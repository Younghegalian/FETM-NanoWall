from __future__ import annotations

import argparse
import json

from nano_sem_domain.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a calibrated SEM image into a corrected nano physical domain."
    )
    parser.add_argument("--config", required=True, help="Path to a JSON domain config.")
    parser.add_argument("--image", required=True, help="Path to the SEM image.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--depth-npy",
        default=None,
        help="Optional precomputed depth map. Skips Depth-Anything-3 inference.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_pipeline(args.config, args.image, args.out, args.depth_npy)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
