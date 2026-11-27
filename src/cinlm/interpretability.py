from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from cinlm.model import CinLM, InterventionSpec


@dataclass
class FrontierAuditReport:
    """Compact, JSON-serializable mechanistic trace of a single prediction,
    explicitly formatted for a Frontier LLM to read, hypothesize circuits,
    and propose causal InterventionSpecs.
    """

    model_name: str
    tokens: List[str]
    target_position: int
    top_predictions: List[Dict[str, Any]]
    logit_lens_by_layer: List[Dict[str, Any]]
    direct_logit_attribution: List[Dict[str, Any]]
    engram_gates: List[Dict[str, Any]]
    top_attention_edges: List[Dict[str, Any]]
    moe_routing: List[Dict[str, Any]]
    relu2_sparsity: List[Dict[str, Any]]
    intervention_applied: Optional[Dict[str, Any]] = None

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)


@torch.no_grad()
def run_frontier_audit(
    model: CinLM,
    input_ids: torch.Tensor,
    decode_token: Callable[[int], str],
    target_pos: int = -1,
    top_k_preds: int = 5,
    top_k_attn: int = 3,
    intervention: Optional[InterventionSpec] = None,
) -> FrontierAuditReport:
    """Run a single-sequence mechanistic audit tailored for Frontier LLM inspection.

    Computes:
    1. Top-k final next-token predictions & probabilities.
    2. Logit Lens at every layer (what the residual stream predicts at each depth).
    3. Exact Direct Logit Attribution (DLA) of the top-1 predicted token across
       Embedding, Engram, Hybrid Attention (SWA vs Global), Shared Expert, and Routed Experts.
    4. Engram context-aware scalar gates per layer/token.
    5. Strongest attention edges targeting `target_pos` (labeled SWA vs Global).
    6. MoE router selections & ReLU² natural sparsity statistics.
    """
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1, "Pass a single sequence (1, T)"
    model.eval()
    seq_len = input_ids.shape[1]
    pos = target_pos if target_pos >= 0 else seq_len + target_pos
    assert 0 <= pos < seq_len

    out = model(input_ids, intervention=intervention, capture_telemetry=True)
    logits = out["logits"][0, pos].float()
    probs = F.softmax(logits, dim=-1)
    top_vals, top_ids = torch.topk(probs, k=min(top_k_preds, logits.numel()))

    top_predictions = [
        {
            "token_id": int(tid),
            "token": decode_token(int(tid)),
            "prob": round(float(p), 5),
            "logit": round(float(logits[tid]), 4),
        }
        for tid, p in zip(top_ids.tolist(), top_vals.tolist())
    ]
    focus_tid = int(top_ids[0].item())

    # Unembedding direction for focus token, scaled by the final RMSNorm factor at `pos`
    tel = out["telemetry"]
    final_res = tel["final_residual"][0, pos].float()
    rms_scale = torch.rsqrt(final_res.pow(2).mean() + model.config.rms_norm_eps)
    unembed_dir = (model.lm_head.weight[focus_tid].float() * model.final_norm.weight.float()) * rms_scale

    def dla_score(vec: torch.Tensor) -> float:
        return round(float(torch.dot(vec.float(), unembed_dir).item()), 4)

    tokens_str = [decode_token(int(t)) for t in input_ids[0].tolist()]

    # Direct Logit Attribution starts with token embedding
    dla_entries: List[Dict[str, Any]] = [
        {
            "component": "token_embedding",
            "layer": -1,
            "logit_contribution": dla_score(tel["emb_out"][0, pos]),
        }
    ]
    logit_lens: List[Dict[str, Any]] = []
    engram_gates: List[Dict[str, Any]] = []
    top_attn_edges: List[Dict[str, Any]] = []
    moe_routing: List[Dict[str, Any]] = []
    relu2_sparsity: List[Dict[str, Any]] = []

    for l_idx, l_tel in enumerate(tel["layers"]):
        # Logit Lens at end of layer
        res_l = l_tel["residual_post"][0, pos]
        lens_logits = model.lm_head(model.final_norm(res_l.unsqueeze(0))).squeeze(0).float()
        lens_probs = F.softmax(lens_logits, dim=-1)
        l_p, l_tid = torch.topk(lens_probs, k=3)
        logit_lens.append(
            {
                "layer": l_idx,
                "focus_token_prob": round(float(lens_probs[focus_tid]), 5),
                "top3": [
                    {"token": decode_token(int(t)), "prob": round(float(p), 4)}
                    for t, p in zip(l_tid.tolist(), l_p.tolist())
                ],
            }
        )

        # Engram attribution & gate
        if "engram_delta" in l_tel:
            eng_score = dla_score(l_tel["engram_delta"][0, pos])
            dla_entries.append(
                {"component": "engram", "layer": l_idx, "logit_contribution": eng_score}
            )
            gate_seq = l_tel["engram_telemetry"]["gate_alpha"][0].tolist()
            engram_gates.append(
                {
                    "layer": l_idx,
                    "gate_at_target": round(float(gate_seq[pos]), 4),
                    "mean_gate_seq": round(sum(gate_seq) / len(gate_seq), 4),
                    "per_token_gates": [
                        {"pos": i, "token": tokens_str[i], "gate": round(float(g), 3)}
                        for i, g in enumerate(gate_seq)
                    ],
                }
            )

        # Attention attribution & top edges
        attn_kind = "global" if l_tel["attn_is_global"] else "swa"
        dla_entries.append(
            {
                "component": f"attn_{attn_kind}",
                "layer": l_idx,
                "logit_contribution": dla_score(l_tel["attn_delta"][0, pos]),
            }
        )
        attn_w = l_tel.get("attn_weights")  # (1, H, T, T)
        if attn_w is not None:
            w_pos = attn_w[0, :, pos, :]  # (H, T)
            head_summaries = []
            for h_idx in range(w_pos.shape[0]):
                k_eff = min(top_k_attn, w_pos.shape[1])
                vals, src_pos = torch.topk(w_pos[h_idx], k=k_eff)
                head_summaries.append(
                    {
                        "head": h_idx,
                        "top_sources": [
                            {
                                "src_pos": int(s),
                                "src_token": tokens_str[int(s)],
                                "weight": round(float(v), 4),
                            }
                            for s, v in zip(src_pos.tolist(), vals.tolist())
                            if float(v) > 0.05
                        ],
                    }
                )
            top_attn_edges.append({"layer": l_idx, "kind": attn_kind, "heads": head_summaries})

        # MoE / Dense MLP attribution & sparsity
        if "moe_telemetry" in l_tel and l_tel["moe_telemetry"] is not None:
            mt = l_tel["moe_telemetry"]
            dla_entries.append(
                {
                    "component": "moe_shared_expert",
                    "layer": l_idx,
                    "logit_contribution": dla_score(mt["shared_out"][0, pos]),
                }
            )
            dla_entries.append(
                {
                    "component": "moe_routed_experts",
                    "layer": l_idx,
                    "logit_contribution": dla_score(mt["routed_out"][0, pos]),
                }
            )
            moe_routing.append(
                {
                    "layer": l_idx,
                    "target_experts": [int(e) for e in mt["topk_experts"][0, pos].tolist()],
                    "target_weights": [round(float(w), 4) for w in mt["topk_weights"][0, pos].tolist()],
                }
            )
            relu2_sparsity.append(
                {
                    "layer": l_idx,
                    "shared_zero_frac": round(float(mt["shared_relu2_zero_frac"][0, pos]), 4),
                    "routed_zero_frac": round(float(mt["routed_relu2_zero_frac"][0, pos]), 4),
                }
            )
        else:
            dla_entries.append(
                {
                    "component": "dense_relu2_mlp",
                    "layer": l_idx,
                    "logit_contribution": dla_score(l_tel["ffn_delta"][0, pos]),
                }
            )
            if l_tel.get("mlp_stats") is not None:
                relu2_sparsity.append(
                    {
                        "layer": l_idx,
                        "dense_zero_frac": round(float(l_tel["mlp_stats"]["relu2_zero_frac"][0, pos]), 4),
                    }
                )

    # Sort DLA by absolute magnitude so Frontier LLM spots top causal drivers immediately
    dla_entries_sorted = sorted(
        dla_entries, key=lambda d: abs(d["logit_contribution"]), reverse=True
    )

    return FrontierAuditReport(
        model_name=model.config.name,
        tokens=tokens_str,
        target_position=pos,
        top_predictions=top_predictions,
        logit_lens_by_layer=logit_lens,
        direct_logit_attribution=dla_entries_sorted,
        engram_gates=engram_gates,
        top_attention_edges=top_attn_edges,
        moe_routing=moe_routing,
        relu2_sparsity=relu2_sparsity,
        intervention_applied=asdict(intervention) if intervention is not None else None,
    )
