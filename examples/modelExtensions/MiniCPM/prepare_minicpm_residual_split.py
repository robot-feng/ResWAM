#!/usr/bin/env python3
"""Create the frozen, source-demo-grouped LIBERO residual split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starVLA.dataloader.minicpm_res_splits import (
    build_residual_split_manifest,
    validate_residual_split_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANNOTATIONS = (
    ROOT / "examples/modelExtensions/MiniCPM/annotations/libero_goal_success_terminals.jsonl"
)
DEFAULT_DATASET = Path(
    "/data/tzq/datasets/starVLA/Datasets/libero_10hz/"
    "libero_goal_no_noops_1.0.0_lerobot"
)
DEFAULT_OUTPUT = (
    ROOT / "examples/modelExtensions/MiniCPM/annotations/"
    "libero_goal_residual_split_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify an existing frozen manifest against its sources without rewriting it",
    )
    args = parser.parse_args()

    if args.verify:
        if not args.output.is_file():
            raise FileNotFoundError(f"frozen split manifest not found: {args.output}")
        manifest = json.loads(args.output.read_text(encoding="utf-8"))
        validate_residual_split_manifest(
            manifest,
            success_terminal_manifest=args.annotations,
            dataset_path=args.dataset,
        )
        print(
            json.dumps(
                {
                    "verified": True,
                    "output": str(args.output),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "counts": manifest["counts"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen split manifest {args.output}; "
            "create a new versioned output path for a new split"
        )
    annotations_path = args.annotations.resolve()
    try:
        annotation_reference = str(annotations_path.relative_to(ROOT))
    except ValueError:
        annotation_reference = str(annotations_path)
    manifest = build_residual_split_manifest(
        success_terminal_manifest=args.annotations,
        dataset_path=args.dataset,
        success_manifest_repo_path=annotation_reference,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validate_residual_split_manifest(
        manifest,
        success_terminal_manifest=args.annotations,
        dataset_path=args.dataset,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": manifest["manifest_sha256"],
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
