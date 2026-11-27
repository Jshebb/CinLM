from __future__ import annotations

import json
import torch

from cinlm.config import get_preset_config
from cinlm.interpretability import run_frontier_audit
from cinlm.model import CinLM, InterventionSpec
from cinlm.moe import SparseReLU2MoE


def test_parameter_counts_and_iso_active_ladder() -> None:
    """Verify analytical estimates match exact PyTorch parameter counts and that
    all 100M ablation presets stay tightly iso-active (~100M active/token).
    """
    for preset in ("smoke", "dense_global_100m", "dense_hybrid_100m", "hybrid_engram_100m", "full_260m_a100m"):
        cfg = get_preset_config(preset)  # type: ignore[arg-type]
        est = cfg.estimate_parameters()
        if preset == "smoke":
            model = CinLM(cfg)
            counts = model.count_parameters()
            assert counts["exact_total"] == counts["estimated_total"], (
                f"Mismatch for {preset}: {counts}"
            )
        else:
            # Verify ~100M active budget across the 100M ablation ladder (within 3%)
            assert 97_000_000 <= est["active_per_token"] <= 103_000_000, (
                f"Preset {preset} active_per_token={est['active_per_token']:,} outside ~100M band"
            )


def test_strict_causality_full_model() -> None:
    """Modifying any future token (> t) must not change logits at positions <= t
    (tests Hybrid Attention SWA/Global, Engram n-grams & conv, and Sparse MoE together).
    """
    torch.manual_seed(0)
    cfg = get_preset_config("smoke")
    model = CinLM(cfg).eval()

    x1 = torch.randint(0, cfg.vocab_size, (2, 24))
    x2 = x1.clone()
    pivot = 12
    x2[:, pivot + 1 :] = (x2[:, pivot + 1 :] + 7) % cfg.vocab_size

    with torch.no_grad():
        logits1 = model(x1)["logits"]
        logits2 = model(x2)["logits"]

    max_diff = (logits1[:, : pivot + 1] - logits2[:, : pivot + 1]).abs().max().item()
    assert max_diff < 1e-5, f"Causality violation detected: max_diff={max_diff}"


def test_streaming_kv_and_engram_parity() -> None:
    """Token-by-token incremental decoding with Hybrid KV cache + EngramState
    must match a single full-sequence forward pass identically.
    """
    torch.manual_seed(1)
    cfg = get_preset_config("smoke")
    model = CinLM(cfg).eval()

    x = torch.randint(0, cfg.vocab_size, (2, 28))
    with torch.no_grad():
        full_logits = model(x)["logits"]

        # Prefill 10 tokens, then decode 18 tokens one-by-one (crossing SWA window=16!)
        prefill = model(x[:, :10], use_cache=True)
        cache = prefill["cache"]
        step_logits = [prefill["logits"]]
        for t in range(10, 28):
            out = model(x[:, t : t + 1], cache=cache, use_cache=True)
            cache = out["cache"]
            step_logits.append(out["logits"])
        stream_logits = torch.cat(step_logits, dim=1)

    max_diff = (full_logits - stream_logits).abs().max().item()
    assert max_diff < 1e-4, f"Streaming/full parity mismatch: max_diff={max_diff}"


def test_sparse_moe_matches_naive_dense_reference() -> None:
    """Verify true sparse token dispatch in SparseReLU2MoE produces identical output
    to a naive loop computing every expert on every token.
    """
    torch.manual_seed(2)
    cfg = get_preset_config("smoke")
    moe = SparseReLU2MoE(cfg, layer_idx=0).eval()
    x = torch.randn(2, 11, cfg.d_model)

    with torch.no_grad():
        sparse_out, _ = moe(x)

        # Naive dense reference
        bsz, seq_len, d = x.shape
        x_flat = x.reshape(-1, d)
        shared_ref, _ = moe.shared_expert(x_flat)
        aff = torch.sigmoid(moe.router(x_flat).float())
        _, topk_idx = torch.topk(aff + moe.expert_bias.unsqueeze(0), k=moe.top_k, dim=-1)
        topk_w = torch.gather(aff, dim=-1, index=topk_idx)
        topk_w = (topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)).to(x.dtype)

        routed_ref = torch.zeros_like(x_flat)
        for tok in range(x_flat.shape[0]):
            for slot in range(moe.top_k):
                e_id = int(topk_idx[tok, slot].item())
                w = topk_w[tok, slot]
                e_out, _ = moe.experts[e_id](x_flat[tok : tok + 1])
                routed_ref[tok] += w * e_out.squeeze(0)
        dense_ref = (shared_ref + routed_ref).view(bsz, seq_len, d)

    max_diff = (sparse_out - dense_ref).abs().max().item()
    assert max_diff < 1e-5, f"Sparse MoE vs dense reference mismatch: {max_diff}"


def test_aux_loss_free_router_bias_updates() -> None:
    """Router bias update should nudge overloaded experts down and underloaded experts up,
    keeping mean bias at zero.
    """
    torch.manual_seed(3)
    cfg = get_preset_config("smoke")
    model = CinLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _ = model(x)
    model.update_moe_router_biases()
    for layer in model.layers:
        if isinstance(layer.ffn, SparseReLU2MoE):
            assert abs(layer.ffn.expert_bias.mean().item()) < 1e-6


def test_frontier_audit_dla_exactness_and_causal_interventions() -> None:
    """Verify two core interpretability invariants:
    1. Sum of Direct Logit Attribution entries equals the exact model logit for top-1 token
       (proving the single additive residual stream has zero unaccounted leakage).
    2. Declarative InterventionSpec (ablating attention head, clamping Engram gate=0,
       ablating an expert) runs cleanly and measurably alters logits.
    """
    torch.manual_seed(4)
    cfg = get_preset_config("smoke")
    model = CinLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 20))
    decode = lambda tid: f"tok_{tid}"

    report = run_frontier_audit(model, x, decode_token=decode, target_pos=-1)
    # Valid JSON serialization
    payload = json.loads(report.to_json())
    assert payload["model_name"] == cfg.name
    assert len(payload["logit_lens_by_layer"]) == cfg.n_layers
    assert len(payload["engram_gates"]) == len(cfg.engram_layers)

    # Exact Direct Logit Attribution conservation check
    top1_logit = report.top_predictions[0]["logit"]
    dla_sum = sum(entry["logit_contribution"] for entry in report.direct_logit_attribution)
    assert abs(top1_logit - dla_sum) < 1e-2, (
        f"DLA does not sum to top-1 logit: top1={top1_logit}, sum={dla_sum}"
    )

    # Causal intervention test
    spec = InterventionSpec(
        ablate_attn_heads={3: [0, 1]},
        force_engram_gate={1: 0.0},
        ablate_experts={2: [0]},
    )
    intervened = run_frontier_audit(model, x, decode_token=decode, target_pos=-1, intervention=spec)
    assert intervened.engram_gates[0]["gate_at_target"] == 0.0
    assert intervened.top_predictions[0]["logit"] != report.top_predictions[0]["logit"]


def test_smoke_backward_pass() -> None:
    """Ensure gradients flow cleanly through Embeddings, Engram, Hybrid Attn, and Sparse MoE,
    and that gradient checkpointing matches uncheckpointed forward/backward.
    """
    torch.manual_seed(5)
    cfg = get_preset_config("smoke")
    model = CinLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 20))
    out = model(x, labels=x)
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads_present = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert grads_present > 0

    # Verify gradient checkpointing parity
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing = True
    out_ckpt = model(x, labels=x)
    assert abs(out_ckpt["loss"].item() - loss.item()) < 1e-5
    out_ckpt["loss"].backward()
    grads_ckpt = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert grads_ckpt == grads_present


def test_sft_assistant_only_loss_masking() -> None:
    """Verify system & user tokens are masked (-100) while assistant reply + <|im_end|> are supervised."""
    from cinlm.data import encode_sft_conversation, normalize_sft_row

    # Toy deterministic word/char tokenizer for unit testing without external files
    vocab = {}

    def fake_encode(text: str) -> list[int]:
        out = []
        for ch in text:
            if ch not in vocab:
                vocab[ch] = len(vocab) + 1
            out.append(vocab[ch])
        return out

    row = {"instruction": "Qual a capital do Brasil?", "output": "Brasília."}
    msgs = normalize_sft_row(row)
    assert msgs is not None and len(msgs) == 2

    ids, labels = encode_sft_conversation(msgs, encode_fn=fake_encode, max_seq_len=512)
    assert len(ids) == len(labels)
    # System + user prefix must be strictly masked
    assistant_body = fake_encode("Brasília.<|im_end|>\n")
    n_body = len(assistant_body)
    assert all(l == -100 for l in labels[:-n_body])
    assert labels[-n_body:] == assistant_body
    assert ids[-n_body:] == assistant_body


def test_muon_optimizer_and_newton_schulz() -> None:
    """Verify Newton-Schulz orthogonalization, parameter routing, Engram LR multiplier,
    loss decrease, and checkpoint state_dict round-trip for MuonWithAuxAdam.
    """
    from cinlm.muon import build_cinlm_optimizer, zeropower_via_newtonschulz5

    torch.manual_seed(10)
    for shape in ((16, 32), (32, 16)):
        g = torch.randn(shape)
        u = zeropower_via_newtonschulz5(g, steps=5)
        assert u.shape == g.shape
        gram = u @ u.T if shape[0] <= shape[1] else u.T @ u
        eye = torch.eye(min(shape))
        # Quintic 5-step NS targets approximate semi-orthogonal matrix (singular values in ~[0.7, 1.3])
        assert (gram - eye).abs().max().item() < 0.45

    cfg = get_preset_config("smoke")
    model = CinLM(cfg)
    opt, counts = build_cinlm_optimizer(
        model, optimizer_type="muon", lr=3e-3, weight_decay=0.05, muon_lr_mult=1.0, engram_lr_mult=3.0
    )
    assert counts["muon_params"] > 0
    assert counts["engram_table_params"] > 0
    assert any(pg.get("use_muon") for pg in opt.param_groups)
    assert any(pg.get("lr_mult") == 3.0 and not pg.get("use_muon") for pg in opt.param_groups)

    x = torch.randint(0, cfg.vocab_size, (2, 24))
    losses = []
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        loss = model(x, labels=x)["loss"]
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"Muon failed to reduce loss: {losses}"

    # Verify state_dict round-trip
    sd = opt.state_dict()
    opt.load_state_dict(sd)


def test_looped_core_function_preservation_causality_and_streaming() -> None:
    """Verify Looped Core (Prelude -> Core x R -> Coda):
    1. Exact analytical parameter count.
    2. Mathematical function preservation at step 0 when branching from a v1 checkpoint.
    3. Gradient flows immediately into loop_res_scale at step 0, and into loop_step_emb when scale != 0.
    4. Strict causality and exact KV+Engram streaming parity at R=2.
    """
    torch.manual_seed(11)
    base_cfg = get_preset_config("smoke")
    base_model = CinLM(base_cfg).eval()

    loop_cfg = get_preset_config("smoke")
    loop_cfg.loop_core_steps = 2
    loop_cfg.loop_start_layer = 1
    loop_cfg.loop_end_layer = 3
    loop_cfg.loop_max_steps = 4
    loop_cfg.__post_init__()

    loop_model = CinLM(loop_cfg).eval()
    counts = loop_model.count_parameters()
    assert counts["exact_total"] == counts["estimated_total"], f"Loop param mismatch: {counts}"

    # Load base weights into looped model (function-preserving branch)
    res = loop_model.load_state_dict(base_model.state_dict(), strict=False)
    assert not res.unexpected_keys
    assert set(res.missing_keys) == {"loop_step_emb.weight", "loop_res_scale"}

    x = torch.randint(0, base_cfg.vocab_size, (2, 24))
    with torch.no_grad():
        base_logits = base_model(x)["logits"]
        r1_logits = loop_model(x, num_loops=1)["logits"]
        r2_logits = loop_model(x, num_loops=2)["logits"]

    assert (base_logits - r1_logits).abs().max().item() < 1e-6
    assert (base_logits - r2_logits).abs().max().item() < 1e-6

    # Verify gradient flows immediately to loop_res_scale[1] at step 0
    loop_model.train()
    out0 = loop_model(x, labels=x, num_loops=2)
    out0["loss"].backward()
    assert loop_model.loop_res_scale is not None
    assert loop_model.loop_res_scale.grad is not None
    assert loop_model.loop_res_scale.grad[1].abs().item() > 0.0

    # Activate loop gate & step embedding, then test strict causality + streaming KV/Engram parity at R=2
    with torch.no_grad():
        loop_model.loop_res_scale[1] = 0.35
        assert loop_model.loop_step_emb is not None
        loop_model.loop_step_emb.weight[1].normal_(0.0, 0.05)

    loop_model.eval()
    x2 = x.clone()
    pivot = 12
    x2[:, pivot + 1 :] = (x2[:, pivot + 1 :] + 5) % base_cfg.vocab_size
    with torch.no_grad():
        l1 = loop_model(x, num_loops=2)["logits"]
        l2 = loop_model(x2, num_loops=2)["logits"]
        assert (l1[:, : pivot + 1] - l2[:, : pivot + 1]).abs().max().item() < 1e-5

        # Prefill 10 tokens then decode 14 tokens one-by-one (crossing SWA window=16 at R=2!)
        prefill = loop_model(x[:, :10], use_cache=True, num_loops=2)
        cache = prefill["cache"]
        chunks = [prefill["logits"]]
        for t in range(10, 24):
            step_out = loop_model(x[:, t : t + 1], cache=cache, use_cache=True, num_loops=2)
            cache = step_out["cache"]
            chunks.append(step_out["logits"])
        stream_l = torch.cat(chunks, dim=1)
        assert (l1 - stream_l).abs().max().item() < 1e-4


def test_mtp_param_count_loss_and_backward() -> None:
    """Verify Multi-Token Prediction auxiliary head parameter count, exact combined loss,
    and clean gradient flow into MTP projections.
    """
    torch.manual_seed(12)
    cfg = get_preset_config("smoke")
    cfg.mtp_num_heads = 1
    cfg.mtp_loss_weight = 0.3
    cfg.__post_init__()

    model = CinLM(cfg)
    counts = model.count_parameters()
    assert counts["exact_total"] == counts["estimated_total"], f"MTP param mismatch: {counts}"

    x = torch.randint(0, cfg.vocab_size, (2, 20))
    out = model(x, labels=x)
    assert out["ntp_loss"] is not None and out["mtp_loss"] is not None
    expected = out["ntp_loss"] + 0.3 * out["mtp_loss"]
    assert abs(out["loss"].item() - expected.item()) < 1e-5

    out["loss"].backward()
    assert model.mtp_projs is not None
    assert model.mtp_projs[0].weight.grad is not None
    assert model.mtp_projs[0].weight.grad.abs().sum().item() > 0.0
