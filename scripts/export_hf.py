#!/usr/bin/env python3
"""Export a trained CinLM .pt checkpoint into a standalone Hugging Face Hub repository:
- model.safetensors (BF16 weights only, ~520 MB instead of ~3.1 GB with AdamW)
- config.json + configuration_cinlm.py + modeling_cinlm.py (trust_remote_code=True)
- tokenizer.json + tokenizer_config.json (PreTrainedTokenizerFast-ready)
- README.md (Hugging Face Model Card with Apache 2.0 + dataset attribution)
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

from cinlm.config import CinLMConfig
from cinlm.model import CinLM


CONFIGURATION_PY = '''from transformers import PretrainedConfig


class CinLMHFConfig(PretrainedConfig):
    model_type = "cinlm"

    def __init__(
        self,
        name: str = "cinlm-full-260m-a100m",
        vocab_size: int = 32768,
        d_model: int = 640,
        n_layers: int = 16,
        n_heads: int = 10,
        n_kv_heads: int = 2,
        max_seq_len: int = 2048,
        rope_theta: float = 100000.0,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        dropout: float = 0.0,
        global_attn_every: int = 4,
        swa_window_size: int = 256,
        use_qk_norm: bool = True,
        use_moe: bool = True,
        d_ff_dense: int = 3072,
        d_shared_expert: int = 1024,
        d_routed_expert: int = 1024,
        n_routed_experts: int = 8,
        moe_top_k: int = 2,
        moe_bias_update_rate: float = 1e-3,
        use_engram: bool = True,
        engram_layers=(1, 5),
        engram_ngram_orders=(2, 3),
        engram_heads_per_order: int = 4,
        engram_head_dim: int = 64,
        engram_table_sizes=(32771, 32779, 32783, 32789, 32797, 32801, 32803, 32831),
        engram_conv_kernel: int = 4,
        init_std: float = 0.02,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.name = name
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.global_attn_every = global_attn_every
        self.swa_window_size = swa_window_size
        self.use_qk_norm = use_qk_norm
        self.use_moe = use_moe
        self.d_ff_dense = d_ff_dense
        self.d_shared_expert = d_shared_expert
        self.d_routed_expert = d_routed_expert
        self.n_routed_experts = n_routed_experts
        self.moe_top_k = moe_top_k
        self.moe_bias_update_rate = moe_bias_update_rate
        self.use_engram = use_engram
        self.engram_layers = list(engram_layers)
        self.engram_ngram_orders = list(engram_ngram_orders)
        self.engram_heads_per_order = engram_heads_per_order
        self.engram_head_dim = engram_head_dim
        self.engram_table_sizes = list(engram_table_sizes)
        self.engram_conv_kernel = engram_conv_kernel
        self.init_std = init_std
'''


MODELING_PY = '''from __future__ import annotations

from typing import Optional

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .cinlm_core.config import CinLMConfig
from .cinlm_core.model import CinLM
from .configuration_cinlm import CinLMHFConfig


class CinLMForCausalLM(PreTrainedModel):
    config_class = CinLMHFConfig
    _tied_weights_keys = ["model.lm_head.weight"]

    def __init__(self, config: CinLMHFConfig):
        super().__init__(config)
        core_cfg = CinLMConfig(
            name=config.name,
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            dropout=config.dropout,
            global_attn_every=config.global_attn_every,
            swa_window_size=config.swa_window_size,
            use_qk_norm=config.use_qk_norm,
            use_moe=config.use_moe,
            d_ff_dense=config.d_ff_dense,
            d_shared_expert=config.d_shared_expert,
            d_routed_expert=config.d_routed_expert,
            n_routed_experts=config.n_routed_experts,
            moe_top_k=config.moe_top_k,
            moe_bias_update_rate=config.moe_bias_update_rate,
            use_engram=config.use_engram,
            engram_layers=tuple(config.engram_layers),
            engram_ngram_orders=tuple(config.engram_ngram_orders),
            engram_heads_per_order=config.engram_heads_per_order,
            engram_head_dim=config.engram_head_dim,
            engram_table_sizes=tuple(config.engram_table_sizes),
            engram_conv_kernel=config.engram_conv_kernel,
            init_std=config.init_std,
        )
        self.model = CinLM(core_cfg)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_emb

    def get_output_embeddings(self):
        return self.model.lm_head

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self.model.lm_head.weight = self.model.tok_emb.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        past_key_values=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        out = self.model(
            input_ids=input_ids,
            labels=labels,
            cache=past_key_values,
            use_cache=use_cache,
        )
        return CausalLMOutputWithPast(
            loss=out["loss"],
            logits=out["logits"],
            past_key_values=out["cache"],
        )
'''


def build_model_card(
    repo_id: str,
    cfg: CinLMConfig,
    meta: Dict[str, Any],
    license_id: str,
    stage: str,
) -> str:
    est = cfg.estimate_parameters()
    datasets_yaml = [
        "Polygl0t/gigaverbo-v2",
        "Polygl0t/gigaverbo-v2-synth",
        "HuggingFaceFW/fineweb-edu",
    ]
    if stage == "sft_ptbr":
        datasets_yaml.append("Polygl0t/gigaverbo-v2-sft")
        if license_id == "cc-by-nc-4.0":
            datasets_yaml.append("dominguesm/Canarim-Instruct-PTBR-Dataset")

    ds_lines = "\n".join(f"- {d}" for d in datasets_yaml)
    return f"""---
license: {license_id}
language:
- pt
- en
tags:
- cinlm
- pt-br
- hybrid-attention
- engram
- moe
- relu2
- mechanistic-interpretability
datasets:
{ds_lines}
---

# {repo_id}

**CinLM** is a homemade Portuguese (PT-BR) Small Language Model combining **Hybrid Attention (3:1 SWA + Global)**, **Engram Conditional Memory**, and **Fine-Grained Sparse `ReLU²` MoE** — engineered from day 0 for **high capability per active parameter (~101M active)** and **native mechanistic auditability by Frontier LLMs**.

## Architecture Summary
- **Total parameters:** {est['total']:,} ({est['total']/1e6:.2f}M)
- **Active parameters / token:** {est['active_per_token']:,} ({est['active_per_token']/1e6:.2f}M)
- **Backbone:** {cfg.n_layers} layers, `d_model={cfg.d_model}`, `n_heads={cfg.n_heads}`, `n_kv_heads={cfg.n_kv_heads}` (GQA), RoPE, QK-RMSNorm, `max_seq_len={cfg.max_seq_len}`
- **Hybrid Attention:** 3:1 Sliding Window (`window={cfg.swa_window_size}`) + Global Causal Attention
- **Engram Conditional Memory:** layers `{list(cfg.engram_layers)}` (`{est['engram_tables']/1e6:.2f}M` n-gram hash table params, scalar context-aware gate $\\alpha_t \\in (0, 1)$)
- **Sparse `ReLU²` MoE:** 1 Shared Expert + {cfg.n_routed_experts} Routed Experts (`top-{cfg.moe_top_k}`), auxiliary-loss-free bias balancing
- **Training checkpoint:** stage=`{stage}`, step=`{meta['step']}`, tokens_seen=`{meta['tokens_seen']:,}`

## Quickstart (`transformers`)
```python
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

repo_id = "{repo_id}"
tok = PreTrainedTokenizerFast.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

inputs = tok("A capital do Brasil é", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.model.generate(inputs["input_ids"], max_new_tokens=64, temperature=0.7, top_k=40)
print(tok.decode(out[0]))
```

## Source Code & Frontier-LLM Audit
Full training, evaluation (Bits-Per-Byte, long-context recall, Direct Logit Attribution, causal ablations) at:
https://github.com/Jshebb/CinLM
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CinLM .pt checkpoint to Hugging Face Hub format")
    parser.add_argument("--checkpoint", required=True, help="Path to CinLM .pt checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Path to cinlm_bpe_32k.json")
    parser.add_argument("--out-dir", required=True, help="Output folder ready for huggingface-cli upload")
    parser.add_argument("--repo-id", default="Jshebb/CinLM-260M-A100M")
    parser.add_argument("--license", default="apache-2.0", choices=["apache-2.0", "cc-by-nc-4.0"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    from safetensors.torch import save_file

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["config"])
    for k in ("engram_layers", "engram_ngram_orders", "engram_table_sizes"):
        if k in cfg_dict and isinstance(cfg_dict[k], list):
            cfg_dict[k] = tuple(cfg_dict[k])
    cfg = CinLMConfig(**cfg_dict)
    meta = {
        "step": int(ckpt.get("step", 0)) + 1,
        "tokens_seen": int(ckpt.get("tokens_seen", 0)),
    }
    stage = str(ckpt.get("stage", "pretrain"))

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    target_dtype = dtype_map[args.dtype]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefix keys with `model.` to match CinLMForCausalLM wrapper and drop tied lm_head duplicate
    raw_sd = ckpt["model"]
    hf_sd: Dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if cfg.tie_word_embeddings and k == "lm_head.weight":
            continue
        if v.is_floating_point():
            hf_sd[f"model.{k}"] = v.detach().to(dtype=target_dtype).contiguous()
        else:
            hf_sd[f"model.{k}"] = v.detach().contiguous()

    sf_path = out_dir / "model.safetensors"
    save_file(hf_sd, str(sf_path), metadata={"format": "pt"})

    # Write HF config.json with auto_map for trust_remote_code=True
    hf_cfg = asdict(cfg)
    for k in ("engram_layers", "engram_ngram_orders", "engram_table_sizes"):
        hf_cfg[k] = list(hf_cfg[k])
    hf_cfg.update(
        {
            "model_type": "cinlm",
            "architectures": ["CinLMForCausalLM"],
            "torch_dtype": "bfloat16" if args.dtype == "bf16" else ("float16" if args.dtype == "fp16" else "float32"),
            "auto_map": {
                "AutoConfig": "configuration_cinlm.CinLMHFConfig",
                "AutoModelForCausalLM": "modeling_cinlm.CinLMForCausalLM",
            },
        }
    )
    (out_dir / "config.json").write_text(json.dumps(hf_cfg, indent=2), encoding="utf-8")

    # Copy tokenizer + write tokenizer_config.json
    shutil.copy2(args.tokenizer, out_dir / "tokenizer.json")
    tok_cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "additional_special_tokens": [
            "<|im_start|>",
            "<|im_end|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
        ],
        "model_max_length": cfg.max_seq_len,
    }
    (out_dir / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2), encoding="utf-8")

    # Bundle self-contained trust_remote_code modules (rewriting `from cinlm.` -> relative imports)
    core_dst = out_dir / "cinlm_core"
    core_dst.mkdir(parents=True, exist_ok=True)
    (core_dst / "__init__.py").write_text("", encoding="utf-8")
    src_pkg = Path(__file__).resolve().parent.parent / "src" / "cinlm"
    for mod in ("config.py", "attention.py", "engram.py", "moe.py", "model.py"):
        code = (src_pkg / mod).read_text(encoding="utf-8")
        code = code.replace("from cinlm.", "from .")
        (core_dst / mod).write_text(code, encoding="utf-8")

    (out_dir / "configuration_cinlm.py").write_text(CONFIGURATION_PY, encoding="utf-8")
    (out_dir / "modeling_cinlm.py").write_text(MODELING_PY, encoding="utf-8")

    card = build_model_card(args.repo_id, cfg, meta, args.license, stage)
    (out_dir / "README.md").write_text(card, encoding="utf-8")

    size_mb = sf_path.stat().st_size / (1024 * 1024)
    print(f"[CinLM HF Export] Exported {args.repo_id} to {out_dir} ({size_mb:.1f} MB safetensors, {args.license})")


if __name__ == "__main__":
    main()
