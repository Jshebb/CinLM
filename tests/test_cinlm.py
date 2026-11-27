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
