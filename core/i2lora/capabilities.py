"""
Capability probe for the i2L (image-to-LoRA) feature.

Mirrors ``core/preview/capabilities.py``: the UI calls this (via
``/api/i2lora/capabilities``) to decide what to show. Unlike live preview, i2L
prediction is a *small* forward pass and runs fine on CPU — the only hard
requirement is Pillow (to read images) plus a predictor checkpoint. A real
SigLIP2 encoder (transformers + weights) is needed only if the checkpoint was
trained with one.

Cheap and side-effect free except for reading a checkpoint's metadata header.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


def _have(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def i2lora_capabilities(checkpoint: Optional[str] = None, siglip_path: Optional[str] = None) -> dict:
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "CPU"
    have_pillow = _have("PIL")
    have_transformers = _have("transformers")

    checkpoint = (checkpoint or "").strip()
    ckpt_present = bool(checkpoint) and os.path.exists(os.path.expanduser(checkpoint))

    encoder = None
    needs_siglip = False
    ckpt_summary = None
    reason = None

    if ckpt_present:
        try:
            from .checkpoint import load_config
            from .layer_spec import spec_from_list, summarize_spec
            cfg = load_config(os.path.expanduser(checkpoint))
            encoder = cfg.encoder
            needs_siglip = encoder in ("siglip", "siglip2")
            ckpt_summary = {
                "encoder": cfg.encoder,
                "rank": cfg.rank,
                "image_embed_dim": cfg.image_embed_dim,
                **summarize_spec(spec_from_list(cfg.layers)),
            }
        except Exception as e:  # malformed / foreign checkpoint
            reason = f"checkpoint unreadable: {e}"
    elif checkpoint:
        reason = f"checkpoint not found: {checkpoint}"
    else:
        reason = "no predictor checkpoint set"

    siglip_ok = (not needs_siglip) or (
        have_transformers and bool(siglip_path) and os.path.exists(os.path.expanduser(siglip_path or ""))
    )

    ready = bool(have_pillow and ckpt_present and reason is None and siglip_ok)
    if ready:
        reason = f"ready on {device}"
    elif reason is None:
        if not have_pillow:
            reason = "Pillow not installed (pip install pillow)"
        elif needs_siglip and not have_transformers:
            reason = "checkpoint needs SigLIP2 but transformers is not installed"
        elif needs_siglip and not siglip_path:
            reason = "checkpoint needs SigLIP2 — set the SigLIP2 model path"
        elif needs_siglip:
            reason = "SigLIP2 model path not found"

    return {
        "cuda": cuda,
        "device": device,
        "torch": torch.__version__,
        "have_pillow": have_pillow,
        "have_transformers": have_transformers,
        "checkpoint_present": ckpt_present,
        "checkpoint_summary": ckpt_summary,
        "encoder": encoder,
        "needs_siglip": needs_siglip,
        "ready": ready,
        "reason": reason,
    }
