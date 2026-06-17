"""
Anima LoRA Editor — i2L (image-to-LoRA) package.

Create a style LoRA directly from one or more reference images in a single
forward pass, instead of training one. This is a faithful, standalone
implementation of i2L from:

    Zhongjie Duan, Yingda Chen, "Compressing Image Style Training into a Single
    Model Forward" (arXiv:2606.13809), ModelScope / Alibaba.

The predictor (a frozen image encoder + a transformer with learnable LoRA
queries + compressed decoding heads) maps reference images to LoRA weights for
the Anima DiT, producing a standard ``.safetensors`` LoRA the rest of this
suite already understands.

IMPORTANT: this is the *inference architecture*, not a trainer. It needs a
trained predictor checkpoint. No i2L predictor has been publicly released for
Anima (the paper releases ones for Z-Image / FLUX.2-klein / Hidream-O1), so a
small random ``demo`` predictor is bundled to make the workflow runnable and
testable — its output is a valid but un-stylised LoRA. See ``demo.py``.

Public surface used by ``app.py`` / ``i2l.py``:

    from core.i2lora import (
        anima_layer_spec, I2LConfig, I2LModel,
        build_encoder, load_predictor, save_predictor,
        predict_lora, predict_gray_lora, predict_lora_from_paths,
        make_metadata, i2lora_capabilities, build_demo_predictor,
    )
"""

from .layer_spec import (
    LoRALayerSpec,
    anima_layer_spec,
    summarize_spec,
    TARGET_SETS,
    DEFAULT_TARGET_SET,
)
from .model import I2LConfig, I2LModel
from .encoder import build_encoder, load_images, gray_image, ReferenceImageEncoder
from .checkpoint import load_predictor, save_predictor, load_config
from .predict import (
    predict_lora,
    predict_gray_lora,
    predict_lora_from_paths,
    make_metadata,
    PredictResult,
)
from .capabilities import i2lora_capabilities
from .demo import build_demo_predictor
from .inject import injected_loras, build_linear_map

__all__ = [
    "LoRALayerSpec",
    "anima_layer_spec",
    "summarize_spec",
    "TARGET_SETS",
    "DEFAULT_TARGET_SET",
    "I2LConfig",
    "I2LModel",
    "build_encoder",
    "load_images",
    "gray_image",
    "ReferenceImageEncoder",
    "load_predictor",
    "save_predictor",
    "load_config",
    "predict_lora",
    "predict_gray_lora",
    "predict_lora_from_paths",
    "make_metadata",
    "PredictResult",
    "i2lora_capabilities",
    "build_demo_predictor",
    "injected_loras",
    "build_linear_map",
]
