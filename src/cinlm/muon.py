from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize a 2D gradient/momentum matrix using 5-step quintic Newton-Schulz iteration.

    Coefficients (3.4445, -4.7750, 2.0315) follow Keller Jordan's Muon & Moonshot AI
    ("Muon is Scalable for LLM Training", arXiv:2502.16982).
    """
    assert G.ndim == 2, f"Newton-Schulz expects a 2D tensor, got shape {tuple(G.shape)}"
    a, b, c = (3.4445, -4.7750, 2.0315)

    # BF16 is fast on CUDA sm>=80; keep FP32 on CPU / older GPUs to avoid slow/inaccurate matmuls
    use_bf16 = G.is_cuda and torch.cuda.is_bf16_supported()
    X = G.to(dtype=torch.bfloat16 if use_bf16 else torch.float32)

    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(dtype=G.dtype)


class MuonWithAuxAdam(Optimizer):
    """Unified optimizer combining Muon (for 2D hidden linear matrices) and AdamW
    (for embeddings, norms, routers, biases, and sparse Engram lookup tables).

    Each param group supports:
      - `use_muon`: bool (True -> Newton-Schulz orthogonalized Nesterov momentum; False -> AdamW)
      - `lr_mult`: float multiplier applied on top of scheduled base LR (e.g., 3.0 for Engram tables)
      - `weight_decay`: decoupled weight decay coefficient
      - `momentum` / `nesterov` / `ns_steps` (Muon groups)
      - `betas` / `eps` (AdamW groups)

    Uses Moonshot AI's RMS-matching factor `0.2 * sqrt(max(m, n))` on Muon updates so Muon
    transfers cleanly at the same base learning rate scale as AdamW (`muon_lr_mult = 1.0`).
    """

    def __init__(self, param_groups: Iterable[Dict[str, Any]]) -> None:
        defaults = dict(
            lr=3e-4,
            lr_mult=1.0,
            weight_decay=0.1,
            use_muon=False,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            use_muon = bool(group["use_muon"])

            if use_muon:
                momentum = float(group["momentum"])
                nesterov = bool(group["nesterov"])
                ns_steps = int(group["ns_steps"])

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.ndim != 2:
                        raise ValueError(f"Muon param group received non-2D tensor {tuple(p.shape)}")

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    update = g.add(buf, alpha=momentum) if nesterov else buf

                    orth = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    # Moonshot RMS-matching scale: matches AdamW update RMS (~0.2)
                    rms_scale = 0.2 * math.sqrt(max(p.size(0), p.size(1)))

                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(orth, alpha=-lr * rms_scale)
            else:
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                    state["step"] += 1
                    step_t = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    g_fp32 = g.float()
                    exp_avg.mul_(beta1).add_(g_fp32, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g_fp32, g_fp32, value=1.0 - beta2)

                    bias_c1 = 1.0 - beta1**step_t
                    bias_c2 = 1.0 - beta2**step_t
                    step_size = lr / bias_c1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_c2)).add_(eps)

                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.add_((exp_avg / denom).to(dtype=p.dtype), alpha=-step_size)

        return loss


def _is_muon_eligible(name: str, p: torch.Tensor) -> bool:
    """Return True for 2D hidden linear weight matrices suitable for Muon orthogonalization."""
    if p.ndim != 2:
        return False
    excluded = ("tok_emb", "lm_head", "tables", "router", "loop_step_emb")
    return not any(tok in name for tok in excluded)


def build_cinlm_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_type: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    muon_lr_mult: float = 1.0,
    engram_lr_mult: float = 1.0,
) -> Tuple[Optimizer, Dict[str, int]]:
    """Build either fused AdamW or Muon+AuxAdamW with optional separate LR multiplier for Engram tables."""
    muon_params: List[torch.Tensor] = []
    adam_decay_params: List[torch.Tensor] = []
    adam_no_decay_params: List[torch.Tensor] = []
    engram_table_params: List[torch.Tensor] = []

    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))

        if "tables" in name:
            engram_table_params.append(p)
        elif optimizer_type == "muon" and _is_muon_eligible(name, p):
            muon_params.append(p)
        elif p.ndim < 2 or "norm" in name or "loop_step_emb" in name or "loop_res_scale" in name:
            adam_no_decay_params.append(p)
        else:
            adam_decay_params.append(p)

    counts = {
        "muon_params": sum(p.numel() for p in muon_params),
        "adam_decay_params": sum(p.numel() for p in adam_decay_params),
        "adam_no_decay_params": sum(p.numel() for p in adam_no_decay_params),
        "engram_table_params": sum(p.numel() for p in engram_table_params),
    }

    if optimizer_type == "adamw":
        groups: List[Dict[str, Any]] = []
        if adam_decay_params:
            groups.append({"params": adam_decay_params, "weight_decay": weight_decay, "lr_mult": 1.0})
        if adam_no_decay_params:
            groups.append({"params": adam_no_decay_params, "weight_decay": 0.0, "lr_mult": 1.0})
        if engram_table_params:
            groups.append({"params": engram_table_params, "weight_decay": 0.0, "lr_mult": engram_lr_mult})
        fused = torch.cuda.is_available()
        opt: Optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=fused)
        return opt, counts

    if optimizer_type == "muon":
        groups = []
        if muon_params:
            groups.append(
                {
                    "params": muon_params,
                    "use_muon": True,
                    "weight_decay": weight_decay,
                    "lr": lr * muon_lr_mult,
                    "lr_mult": muon_lr_mult,
                    "momentum": 0.95,
                    "nesterov": True,
                    "ns_steps": 5,
                }
            )
        if adam_decay_params:
            groups.append(
                {
                    "params": adam_decay_params,
                    "use_muon": False,
                    "weight_decay": weight_decay,
                    "lr": lr,
                    "lr_mult": 1.0,
                }
            )
        if adam_no_decay_params:
            groups.append(
                {
                    "params": adam_no_decay_params,
                    "use_muon": False,
                    "weight_decay": 0.0,
                    "lr": lr,
                    "lr_mult": 1.0,
                }
            )
        if engram_table_params:
            groups.append(
                {
                    "params": engram_table_params,
                    "use_muon": False,
                    "weight_decay": 0.0,
                    "lr": lr * engram_lr_mult,
                    "lr_mult": engram_lr_mult,
                }
            )
        return MuonWithAuxAdam(groups), counts

    raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
