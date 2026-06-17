"""
Differentiable LoRA injection — the piece a *trainer* needs.

At inference the suite bakes a predicted LoRA into the DiT with
``core.anima.lora_anima.merge_to`` (an in-place, bf16 weight add). That is fine
for sampling but useless for *training* an i2L predictor: there is no gradient
path from the merged weights back to the predictor parameters ``phi``.

To meta-train i2L you instead apply the predicted matrices A_l / B_l as an
*additive*, autograd-tracked path on top of the frozen base weights:

        y = W_l x  +  (alpha_l / rank) * (x A_l^T) B_l^T

with ``W_l`` frozen and ``A_l, B_l`` produced by the predictor (so they require
grad). This module does exactly that via forward hooks, keyed by the same LoRA
names the editor uses (``lora_unet_blocks_<N>_<submodule>``), so a predictor
trained this way drops straight into the editor / preview / ComfyUI.

Usage (one training step):

    deltas = predictor(image_tokens)          # {key: {"A","B"}}, requires grad
    with injected_loras(dit, deltas, alpha=rank):
        v_pred = dit(x_sigma, sigma, cross_emb, padding_mask=pad)
    loss = mse(v_pred, v_target)
    loss.backward()                           # grads flow into the predictor

Nothing here is Anima-specific beyond the key-naming convention; the same hook
works on any module whose target layers are ``torch.nn.Linear``.
"""

from __future__ import annotations

import contextlib
from typing import Dict, Optional

import torch
import torch.nn as nn


def build_linear_map(module: nn.Module, prefix: str = "lora_unet") -> Dict[str, nn.Linear]:
    """Map LoRA keys -> the ``nn.Linear`` they target.

    The key for a Linear at dotted path ``blocks.7.cross_attn.k_proj`` is
    ``lora_unet_blocks_7_cross_attn_k_proj`` — identical to how
    ``core.anima.lora_anima`` and ``core.i2lora.layer_spec`` name them.
    """
    out: Dict[str, nn.Linear] = {}
    for name, sub in module.named_modules():
        if isinstance(sub, nn.Linear) and name:
            out[f"{prefix}_{name.replace('.', '_')}"] = sub
    return out


def _lora_hook(A: torch.Tensor, B: torch.Tensor, scale: float):
    """Forward hook adding ``scale * (x A^T) B^T`` to a Linear's output.

    Computed in fp32 so gradients reach the (fp32) predictor outputs cleanly,
    then cast back to the layer's output dtype.
    """
    def hook(_module, args, output):
        x = args[0]
        comp = (x.float() @ A.float().t()) @ B.float().t()
        return output + (scale * comp).to(output.dtype)
    return hook


@contextlib.contextmanager
def injected_loras(
    model: nn.Module,
    deltas: Dict[str, Dict[str, torch.Tensor]],
    alpha: Optional[float] = None,
    module_map: Optional[Dict[str, nn.Linear]] = None,
    strict: bool = True,
):
    """Temporarily add predicted LoRAs to ``model`` for one forward pass.

    Args:
        model:     the frozen backbone (e.g. the Anima DiT).
        deltas:    predictor output ``{key: {"A": (rank,in), "B": (out,rank)}}``.
        alpha:     LoRA alpha; ``None`` -> alpha == rank (gain 1), matching the
                   suite's saved-LoRA convention.
        module_map: optional precomputed key->Linear map (build it once with
                   ``build_linear_map`` and reuse across steps to save work).
        strict:    raise if a predicted key has no matching Linear.

    Yields a small ``{"applied", "missing"}`` dict and removes all hooks on exit
    (even if the forward raises).
    """
    if module_map is None:
        module_map = build_linear_map(model)
    handles = []
    missing = []
    try:
        for key, ab in deltas.items():
            mod = module_map.get(key)
            if mod is None:
                missing.append(key)
                continue
            A, B = ab["A"], ab["B"]
            rank = A.shape[0]
            a = float(alpha if alpha is not None else rank)
            handles.append(mod.register_forward_hook(_lora_hook(A, B, a / rank)))
        if strict and missing:
            raise KeyError(
                f"{len(missing)} predicted layer(s) not found in the model, e.g. "
                f"{missing[:3]}. Check the predictor's layer_spec matches this "
                f"backbone (prefix / submodule names)."
            )
        yield {"applied": len(handles), "missing": missing}
    finally:
        for h in handles:
            h.remove()
