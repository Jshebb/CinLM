#!/usr/bin/env python3
"""Map MoE expert specialization between Direct vs <think> Reasoning modes in CinLM,
and export a declarative Neural Skill Patch (router_bias_patch + causal ablation spec).

Usage:
  python scripts/probe_experts_by_mode.py \
    --checkpoint /path/to/sft_latest.pt \
    --tokenizer /path/to/cinlm_bpe_32k.json \
    --out /path/to/think_skill_patch.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from cinlm.config import CinLMConfig
from cinlm.data import format_chat_example
from cinlm.model import CinLM


DEFAULT_DIRECT_EXAMPLES: List[Tuple[str, str]] = [
    ("Qual é a capital do Brasil?", "A capital do Brasil é Brasília."),
    ("Quem é você?", "Eu sou o CinLM, um assistente de IA aberto em português do Brasil."),
    ("Resuma o que é água.", "A água é uma substância composta por hidrogênio e oxigênio, essencial para a vida na Terra."),
    ("Cite três frutas tropicais.", "Manga, abacaxi e maracujá são três exemplos de frutas tropicais."),
]

DEFAULT_THINK_EXAMPLES: List[Tuple[str, str]] = [
    (
        "Se um trem percorre 60 km em 1 hora e meia, qual é sua velocidade média em km/h?",
        "<think>\nPasso 1: Identificar distância = 60 km e tempo = 1,5 horas.\nPasso 2: Velocidade média = distância / tempo = 60 / 1,5 = 40 km/h.\nPasso 3: Conferir: 40 * 1,5 = 60 km. Correto.\n</think>\n\nA velocidade média do trem é de 40 km/h.",
    ),
    (
        "Ana é mais alta que Bruno, e Bruno é mais alto que Carla. Quem é a pessoa mais baixa?",
        "<think>\nPasso 1: Ana > Bruno.\nPasso 2: Bruno > Carla.\nPasso 3: Logo Ana > Bruno > Carla, então a mais baixa é Carla.\n</think>\n\nA pessoa mais baixa é Carla.",
    ),
    (
        "Uma loja dá 20% de desconto em um livro de R$ 50. Quanto o cliente paga?",
        "<think>\nPasso 1: Calcular 20% de 50 = 0,20 * 50 = 10.\nPasso 2: Subtrair o desconto: 50 - 10 = 40 reais.\nPasso 3: Conferir: 80% de 50 = 40. Correto.\n</think>\n\nO cliente paga R$ 40 pelo livro.",
    ),
    (
        "Todos os mamíferos respiram ar. Baleias são mamíferos. Baleias respiram ar?",
        "<think>\nPasso 1: Premissa 1: todo mamífero respira ar.\nPasso 2: Premissa 2: baleias pertencem ao conjunto dos mamíferos.\nPasso 3: Por silogismo válido, baleias respiram ar.\n</think>\n\nSim, baleias respiram ar porque são mamíferos.",
    ),
]


def _load_model(ckpt_path: str, device: torch.device) -> CinLM:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    for k in ("engram_layers", "engram_ngram_orders", "engram_table_sizes"):
        if k in cfg_dict and isinstance(cfg_dict[k], list):
            cfg_dict[k] = tuple(cfg_dict[k])
    cfg = CinLMConfig(**cfg_dict)
    model = CinLM(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model


@torch.no_grad()
def collect_expert_frequencies(
    model: CinLM,
    tok: Any,
    device: torch.device,
    direct_examples: List[Tuple[str, str]],
    think_examples: List[Tuple[str, str]],
) -> Dict[str, Any]:
    n_layers = model.config.n_layers
    n_experts = model.config.n_routed_experts
    assert model.config.use_moe, "probe_experts_by_mode requires a MoE CinLM model"

    direct_counts = torch.zeros((n_layers, n_experts), dtype=torch.float64)
    think_counts = torch.zeros((n_layers, n_experts), dtype=torch.float64)
    direct_tokens = 0
    think_tokens = 0

    def _accum(prompt_text: str, span_start_char: int, span_end_char: int, target_tensor: torch.Tensor) -> int:
        enc = tok.encode(prompt_text)
        ids = enc.ids[: model.config.max_seq_len]
        offsets = enc.offsets[: model.config.max_seq_len]
        mask = [
            (s >= span_start_char and e <= span_end_char and e > s)
            for (s, e) in offsets
        ]
        n_sel = sum(mask)
        if n_sel == 0:
            return 0
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(x, capture_telemetry=True)
        mask_t = torch.tensor(mask, dtype=torch.bool)
        for layer_idx, l_tel in enumerate(out["telemetry"]["layers"]):
            moe_tel = l_tel.get("moe_telemetry")
            if moe_tel is None:
                continue
            topk_e = moe_tel["topk_experts"][0].cpu()[mask_t]  # (T_sel, K)
            bc = torch.bincount(topk_e.reshape(-1), minlength=n_experts).double()
            target_tensor[layer_idx] += bc
        return n_sel

    for q, a in direct_examples:
        full = format_chat_example([{"role": "user", "content": q}, {"role": "assistant", "content": a}], mode="direct")
        ans_start = full.rfind(a)
        direct_tokens += _accum(full, ans_start, ans_start + len(a), direct_counts)

    for q, a in think_examples:
        full = format_chat_example([{"role": "user", "content": q}, {"role": "assistant", "content": a}], mode="think")
        t_start = full.find("<think>")
        t_end = full.find("</think>") + len("</think>")
        if t_start >= 0 and t_end > t_start:
            think_tokens += _accum(full, t_start, t_end, think_counts)

    direct_probs = direct_counts / direct_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
    think_probs = think_counts / think_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
    diff = think_probs - direct_probs  # positive = specialized for <think>

    per_layer_summary = []
    router_bias_patch: Dict[int, Dict[int, float]] = {}
    top_think_ablation: Dict[int, List[int]] = {}

    for l in range(n_layers):
        vals = diff[l].tolist()
        ranked = sorted(range(n_experts), key=lambda e: vals[e], reverse=True)
        top_think_e = ranked[:2]
        top_direct_e = ranked[-2:]
        per_layer_summary.append(
            {
                "layer": l,
                "top_think_experts": [
                    {
                        "expert": e,
                        "p_think": round(float(think_probs[l, e]), 4),
                        "p_direct": round(float(direct_probs[l, e]), 4),
                        "lift": round(float(vals[e]), 4),
                    }
                    for e in top_think_e
                ],
                "top_direct_experts": [
                    {
                        "expert": e,
                        "p_think": round(float(think_probs[l, e]), 4),
                        "p_direct": round(float(direct_probs[l, e]), 4),
                        "lift": round(float(vals[e]), 4),
                    }
                    for e in top_direct_e
                ],
            }
        )
        # Declarative Neural Skill Patch: boost top positive-lift thinking experts
        layer_patch = {
            int(e): round(min(0.4, max(0.05, float(vals[e]) * 2.0)), 4)
            for e in top_think_e
            if vals[e] > 0.02
        }
        if layer_patch:
            router_bias_patch[l] = layer_patch
            top_think_ablation[l] = [int(top_think_e[0])]

    return {
        "model": model.config.name,
        "direct_tokens_probed": direct_tokens,
        "think_tokens_probed": think_tokens,
        "per_layer": per_layer_summary,
        "declarative_think_skill_patch": {
            "router_bias_patch": router_bias_patch,
            "causal_ablation_targets": top_think_ablation,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe CinLM MoE expert specialization (Direct vs <think>)")
    parser.add_argument("--checkpoint", required=True, help="Path to CinLM .pt checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--out", default="think_skill_patch.json")
    args = parser.parse_args()

    from tokenizers import Tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(args.tokenizer)
    model = _load_model(args.checkpoint, device)

    report = collect_expert_frequencies(model, tok, device, DEFAULT_DIRECT_EXAMPLES, DEFAULT_THINK_EXAMPLES)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[CinLM Expert Probe] Probed {report['direct_tokens_probed']} direct tokens & "
        f"{report['think_tokens_probed']} <think> tokens across {model.config.n_layers} layers."
    )
    print(f"Layers with clear <think>-specialized experts: {len(report['declarative_think_skill_patch']['router_bias_patch'])}")
    print(f"Saved Neural Skill Patch to: {args.out}")


if __name__ == "__main__":
    main()
