"""
Frozen reference-image encoders for i2L.

The paper uses a frozen **SigLIP2** image encoder and keeps the *patch-level*
embeddings (not a single pooled token), because style is distributed across
local texture, palette, composition, and object-independent motifs (Sec. 3.1).
For N reference images the tokens are simply concatenated (eq. 1).

Two encoders are provided behind one tiny interface:

* :class:`SiglipImageEncoder` — the real thing. Loads a SigLIP2 vision model
  through 🤗 ``transformers`` from a user-supplied path and returns its patch
  tokens. Requires ``transformers``/``Pillow`` and the model weights; this is
  what a *trained* i2L predictor expects.

* :class:`FallbackPatchEncoder` — a deterministic, dependency-light stand-in
  (no external weights). It resizes each image, cuts it into a patch grid, and
  applies a fixed seeded random projection. It is **not** SigLIP2 and carries
  no learned semantics — it exists so the whole pipeline is runnable and
  testable (and so the bundled demo predictor produces a structurally valid
  LoRA) on a machine with nothing but torch + Pillow.

Both return a float tensor ``(num_tokens, embed_dim)`` and expose ``embed_dim``
and ``is_real`` so callers can check whether a meaningful encoder is active.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence

import torch


# --- image loading (Pillow optional but recommended) ------------------------

def load_images(paths: Sequence[str]):
    """Load image paths into a list of RGB PIL images.

    Pillow is the recommended dependency for reference-image input (jpg/png/
    webp). We import it lazily and raise a clear, actionable error if absent.
    """
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover - exercised only without Pillow
        raise RuntimeError(
            "Pillow is required to read reference images. Install it with "
            "`pip install pillow` (it ships with the live-preview extras)."
        ) from e
    images = []
    for p in paths:
        with Image.open(p) as im:
            images.append(im.convert("RGB").copy())
    return images


def gray_image(size: int = 224):
    """A pure mid-gray image — the neutral reference for asymmetric guidance
    (Sec. 3.4: the negative CFG branch uses a LoRA predicted from a gray image)."""
    from PIL import Image
    return Image.new("RGB", (size, size), (128, 128, 128))


# --- interface --------------------------------------------------------------

class ReferenceImageEncoder:
    """Common surface: encode PIL images -> ``(num_tokens, embed_dim)`` tokens."""

    embed_dim: int = 0
    is_real: bool = False
    name: str = "base"

    def encode(self, images: List) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


# --- real SigLIP2 encoder ---------------------------------------------------

class SiglipImageEncoder(ReferenceImageEncoder):
    """Frozen SigLIP2 patch-token encoder via 🤗 transformers."""

    is_real = True
    name = "siglip2"

    def __init__(self, model_path: str, device: str = "cpu", dtype=torch.float32):
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                "transformers is required for the SigLIP2 encoder. Install the "
                "preview extras (setup_preview) or `pip install transformers`."
            ) from e
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
        # We only need the vision tower.
        self.vision = getattr(self.model, "vision_model", self.model)
        self.vision.eval().to(device)
        for p in self.vision.parameters():
            p.requires_grad_(False)
        self.embed_dim = int(self.vision.config.hidden_size)

    @torch.no_grad()
    def encode(self, images: List) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        out = self.vision(pixel_values=pixel_values)
        tokens = out.last_hidden_state          # (N, P, embed_dim)
        n, p, d = tokens.shape
        return tokens.reshape(n * p, d).float()  # concat across images (eq. 1)


# --- deterministic fallback encoder (no external weights) -------------------

class FallbackPatchEncoder(ReferenceImageEncoder):
    """Weight-free patch encoder. Deterministic; NOT SigLIP2.

    Pipeline plumbing so the predictor is runnable without a multi-GB vision
    model. Same image -> same tokens (fixed seed), so a demo predictor is
    reproducible.
    """

    is_real = False
    name = "fallback"

    def __init__(self, embed_dim: int = 768, image_size: int = 224, patch: int = 16):
        self.embed_dim = int(embed_dim)
        self.image_size = int(image_size)
        self.patch = int(patch)
        self.grid = self.image_size // self.patch
        self.patch_pixels = self.patch * self.patch * 3
        # Fixed projection (seeded) so encoding is deterministic across runs.
        g = torch.Generator().manual_seed(0xC0FFEE)
        self.proj = torch.randn(self.patch_pixels, self.embed_dim, generator=g)
        self.proj = self.proj / (self.patch_pixels ** 0.5)

    def _to_tensor(self, image) -> torch.Tensor:
        im = image.convert("RGB").resize((self.image_size, self.image_size))
        t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).float() / 255.0
        return t.reshape(self.image_size, self.image_size, 3)

    def encode(self, images: List) -> torch.Tensor:
        tokens = []
        gh = self.grid
        for image in images:
            arr = self._to_tensor(image)                       # (H, W, 3)
            # (grid, patch, grid, patch, 3) -> (grid*grid, patch*patch*3)
            patches = (
                arr.reshape(gh, self.patch, gh, self.patch, 3)
                .permute(0, 2, 1, 3, 4)
                .reshape(gh * gh, self.patch_pixels)
            )
            tokens.append(patches @ self.proj)                 # (P, embed_dim)
        return torch.cat(tokens, dim=0)                        # concat (eq. 1)


# --- factory ----------------------------------------------------------------

def build_encoder(
    encoder: str,
    image_embed_dim: int,
    siglip_path: Optional[str] = None,
    device: str = "cpu",
) -> ReferenceImageEncoder:
    """Construct the encoder a predictor was trained with.

    ``encoder == "siglip2"`` needs ``siglip_path`` (the SigLIP2 weights). The
    ``fallback`` encoder needs nothing and matches ``image_embed_dim`` so its
    tokens line up with the predictor's ``img_in``.
    """
    if encoder in ("siglip2", "siglip"):
        if not siglip_path:
            raise RuntimeError(
                "this predictor expects a SigLIP2 encoder — set the SigLIP2 "
                "model path (siglip_path) to a downloaded SigLIP2 checkpoint."
            )
        enc = SiglipImageEncoder(siglip_path, device=device)
        if enc.embed_dim != image_embed_dim:
            raise RuntimeError(
                f"SigLIP2 embed_dim {enc.embed_dim} != predictor image_embed_dim "
                f"{image_embed_dim} — mismatched encoder for this checkpoint."
            )
        return enc
    if encoder == "fallback":
        return FallbackPatchEncoder(embed_dim=image_embed_dim)
    raise ValueError(f"unknown encoder {encoder!r}")
