"""
The i2L (image-to-LoRA) predictor network.

Faithful to arXiv:2606.13809, "Compressing Image Style Training into a Single
Model Forward" (Duan & Chen). Given patch embeddings of one or more reference
images, the predictor emits a complete LoRA weight set in a single forward
pass — no per-style optimisation.

Architecture (Sec. 3.1, Figure 2):

  1. Image tokens Z_img = Concat(E_img(r_1), ..., E_img(r_N))           (eq. 1)
     come from a frozen image encoder (SigLIP2 in the paper) and are
     projected to the transformer width by ``img_in``.

  2. Learnable LoRA queries Q = {q^A_{l,m}, q^B_{l,m}}: for each of the L
     adapted layers and each of the k ranks, one query generates a *row* of
     A_l and one generates a *column* of B_l. Total 2kL queries.

  3. A single-stream transformer T_phi fuses queries with image tokens:
            H = T_phi([Q ; Z_img])                                      (eq. 3)
     Only the query outputs are decoded.

  4. Compressed linear decoders factor the per-layer head as D_l . C_l so the
     predictor stays compact (Sec. "Compressed linear decoding"):
            A_l[m, :] = D^A_l C^A_l h^A_{l,m}                           (eq. 4)
            B_l[:, m] = D^B_l C^B_l h^B_{l,m}
     C_l reduces dim -> bottleneck c; D_l expands c -> in_l (for A) or out_l
     (for B). There are 2L such decoders.

This module only defines the network and its (de)serialisable config. Turning
the predicted A_l/B_l into a saveable ``.safetensors`` LoRA lives in
``predict.py``; the frozen image encoder lives in ``encoder.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .layer_spec import LoRALayerSpec, spec_from_list, spec_to_list


@dataclass
class I2LConfig:
    """Everything needed to rebuild an :class:`I2LModel` from a checkpoint."""
    dim: int = 768                 # transformer width
    depth: int = 6                 # number of transformer blocks
    heads: int = 12                # attention heads
    rank: int = 16                 # LoRA rank k (queries per matrix per layer)
    bottleneck: int = 64           # compressed-decoder inner width c
    mlp_ratio: float = 4.0
    image_embed_dim: int = 768     # width of the frozen encoder's patch tokens
    encoder: str = "fallback"      # which encoder family this was trained with
    alpha: Optional[float] = None  # default LoRA alpha (None -> alpha == rank)
    layers: List[dict] = field(default_factory=list)  # serialised LoRALayerSpec list

    def layer_specs(self) -> List[LoRALayerSpec]:
        return spec_from_list(self.layers)

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "depth": self.depth,
            "heads": self.heads,
            "rank": self.rank,
            "bottleneck": self.bottleneck,
            "mlp_ratio": self.mlp_ratio,
            "image_embed_dim": self.image_embed_dim,
            "encoder": self.encoder,
            "alpha": self.alpha,
            "layers": self.layers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "I2LConfig":
        return cls(
            dim=int(d.get("dim", 768)),
            depth=int(d.get("depth", 6)),
            heads=int(d.get("heads", 12)),
            rank=int(d.get("rank", 16)),
            bottleneck=int(d.get("bottleneck", 64)),
            mlp_ratio=float(d.get("mlp_ratio", 4.0)),
            image_embed_dim=int(d.get("image_embed_dim", 768)),
            encoder=str(d.get("encoder", "fallback")),
            alpha=(None if d.get("alpha") is None else float(d["alpha"])),
            layers=list(d.get("layers", [])),
        )

    @classmethod
    def for_specs(cls, specs: List[LoRALayerSpec], **kw) -> "I2LConfig":
        cfg = cls(**kw)
        cfg.layers = spec_to_list(specs)
        return cfg


class _TransformerBlock(nn.Module):
    """Pre-norm single-stream transformer block (self-attention + MLP)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class _CompressedDecoder(nn.Module):
    """Factored head D . C: query state (dim) -> a LoRA row/column (out_features).

    ``C`` reduces dim -> bottleneck c; ``D`` expands c -> out_features. Used once
    per adapted layer for A (out_features = in_dim) and once for B
    (out_features = out_dim).
    """

    def __init__(self, dim: int, bottleneck: int, out_features: int):
        super().__init__()
        self.C = nn.Linear(dim, bottleneck, bias=False)
        self.D = nn.Linear(bottleneck, out_features, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.D(self.C(h))


class I2LModel(nn.Module):
    """The image-to-LoRA predictor (the trainable meta-model G_phi)."""

    def __init__(self, config: I2LConfig):
        super().__init__()
        self.config = config
        specs = config.layer_specs()
        if not specs:
            raise ValueError("I2LConfig.layers is empty — nothing to predict")
        self.specs = specs
        L, k, dim = len(specs), config.rank, config.dim

        self.img_in = nn.Linear(config.image_embed_dim, dim)

        # LoRA queries: (L, 2, k, dim). index 0 = A-row queries, 1 = B-col queries.
        self.queries = nn.Parameter(torch.randn(L, 2, k, dim) * 0.02)

        self.blocks = nn.ModuleList(
            _TransformerBlock(dim, config.heads, config.mlp_ratio)
            for _ in range(config.depth)
        )
        self.norm_out = nn.LayerNorm(dim)

        # 2L compressed decoders: A_l -> in_dim rows, B_l -> out_dim columns.
        self.dec_A = nn.ModuleList(
            _CompressedDecoder(dim, config.bottleneck, s.in_dim) for s in specs
        )
        self.dec_B = nn.ModuleList(
            _CompressedDecoder(dim, config.bottleneck, s.out_dim) for s in specs
        )

    # -- introspection ----------------------------------------------------
    @property
    def num_layers(self) -> int:
        return len(self.specs)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- the single forward pass ------------------------------------------
    def forward(self, image_tokens: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Predict every layer's LoRA matrices from reference image tokens.

        Args:
            image_tokens: ``(num_tokens, image_embed_dim)`` patch embeddings,
                already concatenated across all reference images (eq. 1). A
                batch dim is added internally.

        Returns:
            ``{layer_key: {"A": (rank, in_dim), "B": (out_dim, rank)}}``.
        """
        if image_tokens.dim() != 2:
            raise ValueError(
                f"expected image_tokens of shape (num_tokens, embed_dim), "
                f"got {tuple(image_tokens.shape)}"
            )
        L, k, dim = self.num_layers, self.config.rank, self.config.dim
        device = self.queries.device
        dtype = self.queries.dtype
        image_tokens = image_tokens.to(device=device, dtype=dtype)

        z = self.img_in(image_tokens)                       # (Ntok, dim)
        q = self.queries.reshape(L * 2 * k, dim)            # (Lq, dim)
        seq = torch.cat([q, z], dim=0).unsqueeze(0)         # (1, Lq+Ntok, dim)

        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm_out(seq[0])                         # (Lq+Ntok, dim)

        # Recover per-layer query states; the first Lq rows are the queries.
        h = seq[: L * 2 * k].reshape(L, 2, k, dim)

        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for i, spec in enumerate(self.specs):
            hA = h[i, 0]                                     # (k, dim)
            hB = h[i, 1]                                     # (k, dim)
            A = self.dec_A[i](hA)                            # (k, in_dim)
            B = self.dec_B[i](hB).transpose(0, 1)           # (out_dim, k)
            out[spec.key] = {"A": A, "B": B}
        return out
