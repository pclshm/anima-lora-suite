"""
Smoke test for the i2L (image-to-LoRA) feature.

Builds a small demo predictor, predicts a LoRA from a couple of synthetic
reference images, and verifies the result is a valid Anima LoRA that the rest
of the suite accepts: right keys/shapes, detected as ANIMA, analyzable,
editable, and round-trips through save/load. Also checks predictor checkpoint
round-trip, multi-image fusion, and the gray neutral LoRA.

Runs entirely on CPU with just torch + safetensors + Pillow.
"""

import os
import sys
import tempfile

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    load_lora_state_dict,
    save_lora_state_dict,
    detect_architecture,
    analyze_lora,
    edit_lora,
)
from core.detect import summarize_keys, extract_block_info
from core.editor import EditConfig
from core.i2lora import (
    build_demo_predictor,
    build_encoder,
    save_predictor,
    load_predictor,
    predict_lora,
    predict_gray_lora,
    i2lora_capabilities,
    make_metadata,
)


def fake_image(seed, size=64):
    g = torch.Generator().manual_seed(seed)
    arr = (torch.rand(size, size, 3, generator=g) * 255).to(torch.uint8).numpy()
    return Image.fromarray(arr, "RGB")


def main():
    print("=" * 72)
    print(" Anima i2L (image-to-LoRA) — smoke test")
    print("=" * 72)

    # Tiny predictor over the first few blocks so the test is fast.
    model = build_demo_predictor(target_set="attn", blocks=range(3), rank=4, dim=64,
                                 depth=2, heads=4, bottleneck=16)
    print(f"\n[+] Demo predictor: {model.num_layers} layers, "
          f"{model.num_parameters():,} params, rank {model.config.rank}")

    encoder = build_encoder(model.config.encoder, model.config.image_embed_dim)
    assert not encoder.is_real, "demo should use the fallback encoder"

    # --- predict from 2 reference images (multi-image fusion) ---
    images = [fake_image(1), fake_image(2)]
    res = predict_lora(model, encoder, images, dtype=torch.float16)
    sd = res.state_dict
    print(f"[+] Predicted LoRA: {res.info['output_tensor_count']} tensors from "
          f"{res.info['num_reference_images']} images "
          f"({res.info['num_image_tokens']} image tokens)")

    # --- shapes: A is (rank, in), B is (out, rank), alpha scalar ---
    for spec in model.specs:
        A = sd[f"{spec.key}.lora_down.weight"]
        B = sd[f"{spec.key}.lora_up.weight"]
        alpha = sd[f"{spec.key}.alpha"]
        assert A.shape == (model.config.rank, spec.in_dim), (spec.key, A.shape)
        assert B.shape == (spec.out_dim, model.config.rank), (spec.key, B.shape)
        assert alpha.ndim == 0
    print("    A/B/alpha shapes correct for every layer  ✓")

    # --- the suite recognises it as a real Anima LoRA ---
    keys = list(sd.keys())
    arch = detect_architecture(keys)
    print(f"[+] Detected architecture: {arch}")
    assert arch == "ANIMA", f"expected ANIMA, got {arch}"
    summ = summarize_keys(keys)
    assert summ["num_blocks"] == 3, summ
    for k in keys:
        tag, n = extract_block_info(k)
        assert tag == "block" and n in (0, 1, 2), (k, tag, n)
    print(f"    {summ['num_blocks']} blocks, all keys map to AnimaBlocks  ✓")

    # --- analyzer + editor accept the predicted LoRA ---
    imp = analyze_lora(sd)
    assert imp["block_norm"], "analyzer produced no scores"
    edited, einfo = edit_lora(sd, EditConfig(enabled_blocks={0, 2},
                                             llm_adapter_enabled=False,
                                             other_enabled=False))
    assert set(einfo["blocks_kept"]) == {0, 2}, einfo
    for k in edited:
        tag, n = extract_block_info(k)
        assert n in (0, 2)
    print(f"[+] analyze_lora + edit_lora OK (kept blocks {einfo['blocks_kept']})")

    # --- multi-image fusion differs from single-image ---
    res1 = predict_lora(model, encoder, [images[0]], dtype=torch.float16)
    k0 = f"{model.specs[0].key}.lora_up.weight"
    assert not torch.allclose(res1.state_dict[k0].float(), sd[k0].float()), \
        "fusing 2 images should differ from 1 image"
    print("[+] Multi-image fusion changes the prediction  ✓")

    # --- gray neutral LoRA (asymmetric guidance) differs from reference ---
    gray = predict_gray_lora(model, encoder, dtype=torch.float16)
    assert gray.info.get("gray_neutral") is True
    assert not torch.allclose(gray.state_dict[k0].float(), sd[k0].float()), \
        "gray neutral LoRA should differ from the reference LoRA"
    print("[+] Gray neutral LoRA produced and distinct  ✓")

    with tempfile.TemporaryDirectory() as td:
        # --- LoRA file round-trips ---
        lora_path = os.path.join(td, "style.safetensors")
        save_lora_state_dict(sd, lora_path, metadata=make_metadata(res.info, "demo", []))
        reloaded = load_lora_state_dict(lora_path)
        assert set(reloaded.keys()) == set(sd.keys())
        for k in sd:
            assert torch.allclose(sd[k].float(), reloaded[k].float()), k
        print(f"[+] LoRA save/load round-trip OK ({len(reloaded)} tensors)")

        # --- predictor checkpoint round-trips and reproduces the prediction ---
        ckpt = os.path.join(td, "predictor.safetensors")
        save_predictor(model, ckpt)
        model2 = load_predictor(ckpt)
        res2 = predict_lora(model2, encoder, images, dtype=torch.float16)
        for k in sd:
            assert torch.allclose(sd[k].float(), res2.state_dict[k].float(), atol=1e-3), k
        print("[+] Predictor checkpoint round-trip reproduces the LoRA  ✓")

        # --- capabilities probe sees the checkpoint as ready (CPU is fine) ---
        caps = i2lora_capabilities(checkpoint=ckpt)
        print(f"[+] Capabilities: ready={caps['ready']} encoder={caps['encoder']} "
              f"reason={caps['reason']!r}")
        assert caps["checkpoint_present"] is True
        assert caps["ready"] is True, caps
        assert caps["needs_siglip"] is False

    print("\n  All checks passed ✓\n")


if __name__ == "__main__":
    main()
