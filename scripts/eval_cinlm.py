#!/usr/bin/env python3
"""Comprehensive evaluation suite for CinLM checkpoints:
1. PT-BR qualitative text generation (greedy + sampled)
2. Long-context associative recall benchmark crossing the SWA window (128 -> 1536 tokens)
   with counterfactual value swap check
3. Mechanistic / Frontier-LLM interpretability audit + causal ablations (Engram & Global Attn)
4. MoE expert load balance & ReLU² natural activation sparsity report
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from cinlm.config import CinLMConfig, get_preset_config
from cinlm.interpretability import run_frontier_audit
from cinlm.model import CinLM, InterventionSpec
from cinlm.moe import SparseReLU2MoE


PTBR_PROMPTS: List[str] = [
    "A capital do Brasil é",
    "A inteligência artificial moderna funciona através de",
    "Em uma pequena cidade do interior de Minas Gerais, havia",
    "Para explicar a fotossíntese de forma simples, podemos dizer que as plantas",
]

DISTRACTOR_SENTENCES: List[str] = [
    "Durante o século passado, diversas transformações econômicas moldaram o desenvolvimento urbano no país.",
    "Os estudos sobre biodiversidade mostram que florestas tropicais abrigam milhares de espécies ainda pouco conhecidas.",
    "Na física clássica, o movimento dos corpos pode ser descrito por equações diferenciais bem estabelecidas.",
    "A culinária regional brasileira combina ingredientes indígenas, africanos e europeus em pratos muito variados.",
    "Pesquisadores de astronomia utilizam telescópios espaciais para observar galáxias distantes no universo.",
    "A música popular evoluiu ao longo das décadas incorporando novos ritmos, instrumentos e harmonias.",
]


def load_checkpoint(
    checkpoint_path: str | None,
    preset: str,
    device: torch.device,
) -> Tuple[CinLM, Dict[str, Any]]:
    meta: Dict[str, Any] = {"checkpoint": checkpoint_path, "step": 0, "tokens_seen": 0}
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg_dict = dict(ckpt["config"])
        for k in ("engram_layers", "engram_ngram_orders", "engram_table_sizes"):
            if k in cfg_dict and isinstance(cfg_dict[k], list):
                cfg_dict[k] = tuple(cfg_dict[k])
        cfg = CinLMConfig(**cfg_dict)
        model = CinLM(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        meta["step"] = int(ckpt.get("step", 0)) + 1
        meta["tokens_seen"] = int(ckpt.get("tokens_seen", 0))
    else:
        cfg = get_preset_config(preset)  # type: ignore[arg-type]
        model = CinLM(cfg).to(device)
    model.eval()
    return model, meta


@torch.no_grad()
def eval_ptbr_generation(
    model: CinLM,
    tok: Tokenizer,
    device: torch.device,
    max_new_tokens: int = 80,
) -> List[Dict[str, str]]:
    eos_id = tok.token_to_id("<|eos|>")
    results = []
    for prompt in PTBR_PROMPTS:
        ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
        out_greedy = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0, eos_token_id=eos_id)
        out_sample = model.generate(
            ids, max_new_tokens=max_new_tokens, temperature=0.75, top_k=40, eos_token_id=eos_id
        )
        results.append(
            {
                "prompt": prompt,
                "greedy": tok.decode(out_greedy[0].tolist()),
                "sampled": tok.decode(out_sample[0].tolist()),
            }
        )
    return results


@torch.no_grad()
def eval_associative_recall(
    model: CinLM,
    tok: Tokenizer,
    device: torch.device,
    distances: Tuple[int, ...] = (128, 384, 768, 1280),
) -> List[Dict[str, Any]]:
    """Needle-in-a-haystack PT-BR associative recall across SWA window boundaries,
    including counterfactual swap margin (checks whether model truly tracks in-context values).
    """
    pairs = [
        ("Alfa", "azul", "verde"),
        ("Beta", "ouro", "prata"),
        ("Gama", "norte", "sul"),
        ("Delta", "fogo", "gelo"),
    ]
    max_len = model.config.max_seq_len
    summary = []

    for target_dist in distances:
        if target_dist + 96 >= max_len:
            continue
        hits = 0
        swap_flips = 0
        margins = []
        total = 0

        for key, val_a, val_b in pairs:
            # Verify candidate first tokens are distinct
            tid_a = tok.encode(f" {val_a}").ids
            tid_b = tok.encode(f" {val_b}").ids
            if not tid_a or not tid_b or tid_a[0] == tid_b[0]:
                continue
            ca, cb = tid_a[0], tid_b[0]

            # Build distractor haystack to reach ~target_dist tokens between fact and query
            filler = []
            idx = 0
            while len(tok.encode(" ".join(filler)).ids) < target_dist:
                filler.append(DISTRACTOR_SENTENCES[idx % len(DISTRACTOR_SENTENCES)])
                idx += 1
            haystack = " ".join(filler)

            prompt_a = (
                f"Registro: a palavra-chave do projeto {key} é {val_a}. "
                f"A palavra-chave de controle é {val_b}. "
                f"{haystack} "
                f"Pergunta: qual é a palavra-chave do projeto {key}? Resposta:"
            )
            prompt_b = (
                f"Registro: a palavra-chave do projeto {key} é {val_b}. "
                f"A palavra-chave de controle é {val_a}. "
                f"{haystack} "
                f"Pergunta: qual é a palavra-chave do projeto {key}? Resposta:"
            )

            ids_a = tok.encode(prompt_a).ids[-max_len:]
            ids_b = tok.encode(prompt_b).ids[-max_len:]
            logits_a = model(torch.tensor([ids_a], dtype=torch.long, device=device))["logits"][0, -1].float()
            logits_b = model(torch.tensor([ids_b], dtype=torch.long, device=device))["logits"][0, -1].float()

            margin_a = float((logits_a[ca] - logits_a[cb]).item())
            margin_b = float((logits_b[cb] - logits_b[ca]).item())
            if margin_a > 0:
                hits += 1
            if margin_a > 0 and margin_b > 0:
                swap_flips += 1
            margins.append(0.5 * (margin_a + margin_b))
            total += 1

        if total > 0:
            summary.append(
                {
                    "approx_distance_tokens": target_dist,
                    "beyond_swa_window": target_dist > model.config.swa_window_size,
                    "candidate_acc": round(hits / total, 4),
                    "counterfactual_swap_acc": round(swap_flips / total, 4),
                    "mean_logit_margin": round(sum(margins) / len(margins), 4),
                    "n_probes": total,
                }
            )
    return summary


@torch.no_grad()
def eval_mechanistic_health(
    model: CinLM,
    tok: Tokenizer,
    device: torch.device,
    audit_prompt: str = "A capital do Brasil é",
) -> Dict[str, Any]:
    """Run Frontier-LLM audit + causal ablations (Engram off, Global Attention off) + MoE balance."""
    ids = torch.tensor([tok.encode(audit_prompt).ids], dtype=torch.long, device=device)
    decode = lambda tid: tok.decode([tid])

    clean = run_frontier_audit(model, ids, decode_token=decode, target_pos=-1)
    top1_tid = clean.top_predictions[0]["token_id"]
    top1_prob_clean = clean.top_predictions[0]["prob"]

    ablations: Dict[str, Any] = {}
    if model.config.use_engram:
        no_eng = run_frontier_audit(
            model,
            ids,
            decode_token=decode,
            target_pos=-1,
            intervention=InterventionSpec(
                force_engram_gate={l: 0.0 for l in model.config.engram_layers}
            ),
        )
        p_no_eng = next((p["prob"] for p in no_eng.top_predictions if p["token_id"] == top1_tid), 0.0)
        ablations["engram_zeroed"] = {
            "top1_after": no_eng.top_predictions[0],
            "orig_top1_prob_after": p_no_eng,
            "prob_drop": round(top1_prob_clean - p_no_eng, 5),
        }

    global_layers = [i for i in range(model.config.n_layers) if model.config.is_global_layer(i)]
    if global_layers:
        no_glob = run_frontier_audit(
            model,
            ids,
            decode_token=decode,
            target_pos=-1,
            intervention=InterventionSpec(
                ablate_attn_heads={l: list(range(model.config.n_heads)) for l in global_layers}
            ),
        )
        p_no_glob = next((p["prob"] for p in no_glob.top_predictions if p["token_id"] == top1_tid), 0.0)
        ablations["global_attn_zeroed"] = {
            "top1_after": no_glob.top_predictions[0],
            "orig_top1_prob_after": p_no_glob,
            "prob_drop": round(top1_prob_clean - p_no_glob, 5),
        }

    # Summarize ReLU² sparsity and MoE router bias spread
    sparsities = []
    for row in clean.relu2_sparsity:
        if "routed_zero_frac" in row:
            sparsities.append(0.5 * (row["shared_zero_frac"] + row["routed_zero_frac"]))
        elif "dense_zero_frac" in row:
            sparsities.append(row["dense_zero_frac"])

    moe_bias_stats = []
    for l_idx, layer in enumerate(model.layers):
        if isinstance(layer.ffn, SparseReLU2MoE):
            b = layer.ffn.expert_bias.float()
            moe_bias_stats.append(
                {
                    "layer": l_idx,
                    "bias_min": round(float(b.min().item()), 4),
                    "bias_max": round(float(b.max().item()), 4),
                }
            )

    return {
        "audit_prompt": audit_prompt,
        "mean_relu2_zero_frac": round(sum(sparsities) / max(1, len(sparsities)), 4),
        "ablations": ablations,
        "moe_bias_spread": moe_bias_stats,
        "frontier_audit_json": json.loads(clean.to_json()),
    }


PTBR_VAL_TEXTS: List[str] = [
    (
        "A Amazônia é a maior floresta tropical do mundo e desempenha um papel fundamental na regulação "
        "do clima global. Seus rios formam a maior bacia hidrográfica do planeta, concentrando uma enorme "
        "diversidade biológica que inclui milhões de espécies de plantas, insetos, aves e mamíferos."
    ),
    (
        "Na ciência da computação, algoritmos eficientes permitem resolver problemas complexos utilizando "
        "menos memória e menos tempo de processamento. Estruturas de dados como tabelas hash, árvores "
        "balanceadas e grafos são ferramentas essenciais para organizar informações em larga escala."
    ),
    (
        "A história da literatura brasileira atravessa diversos movimentos estéticos, do barroco e do "
        "romantismo ao realismo de Machado de Assis e ao modernismo de 1922. Cada período refletiu as "
        "transformações sociais, políticas e culturais do país em sua época."
    ),
    (
        "O sistema solar é composto pelo Sol e por todos os corpos celestes que orbitam ao seu redor, "
        "incluindo oito planetas, dezenas de luas, asteroides e cometas. A gravidade mantém esses objetos "
        "em trajetórias previsíveis estudadas há séculos pela astronomia."
    ),
]


@torch.no_grad()
def eval_bits_per_byte(
    model: CinLM,
    tok: Tokenizer,
    device: torch.device,
    texts: List[str],
) -> Dict[str, Any]:
    """Compute tokenizer-independent Bits-Per-Byte (BPB) and token perplexity on PT-BR texts.

    Formula:
        BPB = (sum of NLL in nats across predicted tokens) / (ln(2) * total UTF-8 bytes)
    Enables fair comparison against models with different tokenizers (e.g., Gemma-3-270M, SmolLM2).
    """
    total_nll_nats = 0.0
    total_tokens = 0
    total_bytes = 0
    max_len = model.config.max_seq_len

    for text in texts:
        text = text.strip()
        if not text:
            continue
        ids = tok.encode(text).ids[:max_len]
        if len(ids) < 2:
            continue
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(inp)["logits"][:, :-1, :].float()
        targets = inp[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), targets.reshape(-1), reduction="sum")
        total_nll_nats += float(nll.item())
        total_tokens += int(targets.numel())
        # Count UTF-8 bytes covered by the predicted slice (excluding the unpredicted first token)
        decoded_pred = tok.decode(ids[1:])
        total_bytes += max(1, len(decoded_pred.encode("utf-8")))

    mean_nll = total_nll_nats / max(1, total_tokens)
    bpb = total_nll_nats / (math.log(2.0) * max(1, total_bytes))
    return {
        "n_docs": len(texts),
        "total_tokens": total_tokens,
        "total_utf8_bytes": total_bytes,
        "mean_ce_nats": round(mean_nll, 5),
        "token_perplexity": round(math.exp(min(mean_nll, 20.0)), 4),
        "bits_per_byte": round(bpb, 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CinLM checkpoint (PT-BR + BPB + Recall + Frontier Audit)")
    parser.add_argument("--checkpoint", default=None, help="Path to CinLM .pt checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--val-file", default=None, help="Optional UTF-8 text/jsonl validation file for BPB")
    parser.add_argument("--preset", default="smoke", help="Fallback preset if no checkpoint provided")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--out", default="eval_report.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(args.tokenizer)
    model, meta = load_checkpoint(args.checkpoint, args.preset, device)

    if args.val_file:
        raw = Path(args.val_file).read_text(encoding="utf-8").splitlines()
        val_texts = []
        for line in raw:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    val_texts.append(str(json.loads(line).get("text", "")))
                    continue
                except Exception:
                    pass
            val_texts.append(line)
    else:
        val_texts = PTBR_VAL_TEXTS

    print(f"[CinLM Eval] model={model.config.name} step={meta['step']} tokens={meta['tokens_seen']:,}")
    bpb_metrics = eval_bits_per_byte(model, tok, device, val_texts)
    gens = eval_ptbr_generation(model, tok, device, max_new_tokens=args.max_new_tokens)
    recall = eval_associative_recall(model, tok, device)
    mech = eval_mechanistic_health(model, tok, device)

    print("\n=== Validação Independente PT-BR (BPB) ===")
    print(json.dumps(bpb_metrics, ensure_ascii=False, indent=2))

    print("\n=== Amostras PT-BR (sampled) ===")
    for g in gens:
        print(f">>> {g['prompt']}\n{g['sampled']}\n")

    print("=== Recall Associativo (atravessando janela SWA) ===")
    for r in recall:
        print(json.dumps(r, ensure_ascii=False))

    print("\n=== Saúde Mecanística & Ablações ===")
    print(f"ReLU² esparsidade média (zeros): {mech['mean_relu2_zero_frac']*100:.1f}%")
    print("Ablações:", json.dumps(mech["ablations"], ensure_ascii=False, indent=2))

    report = {
        "meta": meta,
        "validation_bpb": bpb_metrics,
        "generations": gens,
        "associative_recall": recall,
        "mechanistic": mech,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRelatório completo salvo em: {args.out}")


if __name__ == "__main__":
    main()
