"""
Turn reference images into a saveable Anima LoRA, in one forward pass.

This is the user-facing entry point of the i2L feature. It wires together the
frozen image encoder (``encoder.py``), the predictor (``model.py``), and the
LoRA key conventions (``layer_spec.py``) to produce a standard ``.safetensors``
LoRA that the editor, analyzer, live preview, and ComfyUI all understand.

A predicted layer's matrices map to LoRA tensors as:

    <key>.lora_down.weight  = A   (rank, in_dim)     # the "down"/A side
    <key>.lora_up.weight    = B   (out_dim, rank)    # the "up"/B side
    <key>.alpha             = alpha                  # scaling (eq. 2)

By convention the rest of the suite scales the *up* side and treats
``alpha/rank`` as the effective gain, so we default ``alpha == rank`` (gain 1)
and fold any requested ``multiplier`` into the B matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .encoder import ReferenceImageEncoder, build_encoder, gray_image, load_images
from .model import I2LModel


@dataclass
class PredictResult:
    state_dict: Dict[str, torch.Tensor]
    info: dict


def _assemble_state_dict(
    deltas: Dict[str, Dict[str, torch.Tensor]],
    alpha: float,
    multiplier: float,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for key, ab in deltas.items():
        A = ab["A"].detach()
        B = ab["B"].detach()
        if multiplier != 1.0:
            B = B * float(multiplier)   # fold gain into the up/B side only
        sd[f"{key}.lora_down.weight"] = A.to(dtype).contiguous()
        sd[f"{key}.lora_up.weight"] = B.to(dtype).contiguous()
        sd[f"{key}.alpha"] = torch.tensor(float(alpha), dtype=torch.float32)
    return sd


@torch.no_grad()
def predict_lora(
    model: I2LModel,
    encoder: ReferenceImageEncoder,
    images: List,
    multiplier: float = 1.0,
    alpha: Optional[float] = None,
    dtype: torch.dtype = torch.float16,
) -> PredictResult:
    """Predict a LoRA state_dict from already-loaded PIL ``images``.

    Multiple images are fused into one LoRA by concatenating their tokens
    (eq. 1 / Sec. 4.5 "Fusing styles from multiple images").
    """
    if not images:
        raise ValueError("need at least one reference image")
    tokens = encoder.encode(images)
    deltas = model(tokens)
    a = float(alpha if alpha is not None else (model.config.alpha or model.config.rank))
    sd = _assemble_state_dict(deltas, a, multiplier, dtype)
    info = {
        "num_reference_images": len(images),
        "num_layers": model.num_layers,
        "rank": model.config.rank,
        "alpha": a,
        "multiplier": multiplier,
        "encoder": encoder.name,
        "encoder_is_real": encoder.is_real,
        "num_image_tokens": int(tokens.shape[0]),
        "output_tensor_count": len(sd),
    }
    return PredictResult(state_dict=sd, info=info)


@torch.no_grad()
def predict_gray_lora(
    model: I2LModel,
    encoder: ReferenceImageEncoder,
    multiplier: float = 1.0,
    alpha: Optional[float] = None,
    dtype: torch.dtype = torch.float16,
) -> PredictResult:
    """Predict the neutral *gray-image* LoRA for asymmetric guidance (Sec. 3.4).

    Used on the negative CFG branch so the guidance direction emphasises the
    style introduced by the reference LoRA rather than generic denoising.
    """
    res = predict_lora(model, encoder, [gray_image()], multiplier, alpha, dtype)
    res.info["gray_neutral"] = True
    return res


def predict_lora_from_paths(
    checkpoint: str,
    image_paths: List[str],
    siglip_path: Optional[str] = None,
    device: str = "cpu",
    multiplier: float = 1.0,
    alpha: Optional[float] = None,
    also_gray: bool = False,
):
    """Full path-based pipeline: load predictor + encoder, read images, predict.

    Returns ``(PredictResult, Optional[PredictResult_gray])``. The optional
    second result is the gray neutral LoRA (when ``also_gray``).
    """
    from .checkpoint import load_predictor

    model = load_predictor(checkpoint, device=device)
    encoder = build_encoder(
        model.config.encoder, model.config.image_embed_dim,
        siglip_path=siglip_path, device=device,
    )
    images = load_images(image_paths)
    main = predict_lora(model, encoder, images, multiplier=multiplier, alpha=alpha)
    gray = predict_gray_lora(model, encoder, multiplier=multiplier, alpha=alpha) if also_gray else None
    return main, gray


def make_metadata(info: dict, checkpoint: str, image_paths: List[str]) -> Dict[str, str]:
    """LoRA-file metadata recording i2L provenance (mirrors the editor's style)."""
    md = {
        "anima_lora_editor": "1",
        "anima_lora_editor.source": "i2lora",
        "i2lora": "1",
        "i2lora.checkpoint": checkpoint,
        "i2lora.num_reference_images": str(info.get("num_reference_images", 0)),
        "i2lora.rank": str(info.get("rank", "")),
        "i2lora.alpha": str(info.get("alpha", "")),
        "i2lora.encoder": str(info.get("encoder", "")),
        "i2lora.encoder_is_real": str(info.get("encoder_is_real", False)).lower(),
    }
    if info.get("gray_neutral"):
        md["i2lora.gray_neutral"] = "true"
    return md
