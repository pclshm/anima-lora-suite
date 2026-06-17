"""
i2L — create an Anima style LoRA from reference images, in one forward pass.

A command-line companion to the editor's web UI for the image-to-LoRA feature
(arXiv:2606.13809). Point it at a trained predictor checkpoint and a handful of
reference images and it writes a standard Anima ``.safetensors`` LoRA.

Examples
--------
  # Build the bundled demo predictor (random weights — see note below)
  python i2l.py make-demo demo_predictor.safetensors

  # Predict a LoRA from images
  python i2l.py create \
      --checkpoint demo_predictor.safetensors \
      --images ref1.png ref2.jpg \
      --output style_lora.safetensors \
      --gray                 # also write the neutral gray LoRA (asymmetric CFG)

  # Inspect what a checkpoint / environment can do
  python i2l.py caps --checkpoint demo_predictor.safetensors

NOTE: there is no publicly released i2L predictor for Anima. ``make-demo``
creates a *random* predictor so you can exercise the full workflow; its LoRA is
structurally valid but not trained, so it won't stylise like a real predictor.
Drop in a trained checkpoint for real results.
"""

import argparse
import json
import os
import sys

from core.i2lora import (
    build_demo_predictor,
    save_predictor,
    predict_lora_from_paths,
    make_metadata,
    i2lora_capabilities,
    TARGET_SETS,
    DEFAULT_TARGET_SET,
)
from core.editor import save_lora_state_dict


def _gray_path(output: str) -> str:
    root, ext = os.path.splitext(output)
    return f"{root}.gray{ext or '.safetensors'}"


def cmd_make_demo(args):
    blocks = range(args.blocks) if args.blocks else None
    model = build_demo_predictor(target_set=args.target_set, blocks=blocks,
                                 rank=args.rank)
    save_predictor(model, args.output, extra_metadata={"i2lora.demo": "true"})
    print(f"[+] Wrote demo predictor -> {args.output}")
    print(f"    {model.num_layers} layers, {model.num_parameters():,} params, "
          f"rank {model.config.rank}, target_set={args.target_set}")
    print("    NOTE: random weights — for workflow/validation, not quality.")


def cmd_create(args):
    if not os.path.exists(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint}")
    missing = [p for p in args.images if not os.path.exists(p)]
    if missing:
        sys.exit("reference image(s) not found: " + ", ".join(missing))

    device = "cuda" if args.device == "cuda" else "cpu"
    main, gray = predict_lora_from_paths(
        checkpoint=args.checkpoint,
        image_paths=args.images,
        siglip_path=args.siglip,
        device=device,
        multiplier=args.multiplier,
        alpha=args.alpha,
        also_gray=args.gray,
    )

    md = make_metadata(main.info, args.checkpoint, args.images)
    save_lora_state_dict(main.state_dict, args.output, metadata=md)
    print(f"[+] Wrote LoRA -> {args.output}")
    print("    " + json.dumps(main.info))
    if not main.info["encoder_is_real"]:
        print("    NOTE: fallback encoder (not SigLIP2) — output is not "
              "meaningfully stylised. Use a SigLIP2-trained checkpoint + "
              "--siglip for real results.")

    if gray is not None:
        gp = _gray_path(args.output)
        save_lora_state_dict(gray.state_dict, gp, metadata=make_metadata(gray.info, args.checkpoint, []))
        print(f"[+] Wrote gray neutral LoRA -> {gp}")


def cmd_caps(args):
    caps = i2lora_capabilities(checkpoint=args.checkpoint, siglip_path=args.siglip)
    print(json.dumps(caps, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("make-demo", help="write a random demo predictor checkpoint")
    d.add_argument("output")
    d.add_argument("--target-set", default=DEFAULT_TARGET_SET, choices=sorted(TARGET_SETS))
    d.add_argument("--blocks", type=int, default=0, help="limit to first N blocks (0 = all 28)")
    d.add_argument("--rank", type=int, default=8)
    d.set_defaults(func=cmd_make_demo)

    c = sub.add_parser("create", help="predict a LoRA from reference images")
    c.add_argument("--checkpoint", required=True, help="i2L predictor .safetensors")
    c.add_argument("--images", required=True, nargs="+", help="reference image paths")
    c.add_argument("--output", required=True, help="output LoRA .safetensors")
    c.add_argument("--siglip", default=None, help="SigLIP2 model path (if the checkpoint needs it)")
    c.add_argument("--multiplier", type=float, default=1.0, help="LoRA gain (folded into up/B side)")
    c.add_argument("--alpha", type=float, default=None, help="LoRA alpha (default: rank)")
    c.add_argument("--gray", action="store_true", help="also write the gray neutral LoRA")
    c.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    c.set_defaults(func=cmd_create)

    s = sub.add_parser("caps", help="probe i2L capabilities for a checkpoint/env")
    s.add_argument("--checkpoint", default=None)
    s.add_argument("--siglip", default=None)
    s.set_defaults(func=cmd_caps)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
