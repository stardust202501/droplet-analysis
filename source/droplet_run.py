"""Command-line entry point for single-image droplet analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from droplet_analysis import AnalysisConfig, analyze_single_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze fluorescence droplet images.")
    parser.add_argument("input_image", type=Path)
    parser.add_argument("--output-folder", type=Path, default=Path("droplet_results"))
    parser.add_argument("--diameter", type=float, default=40.0)
    args = parser.parse_args()

    config = AnalysisConfig(diameter_px=args.diameter)

    summary = analyze_single_image(
        args.input_image,
        args.output_folder,
        config=config,
    )
    errors = int((summary.get("status") == "error").sum()) if not summary.empty else 0
    print(f"Processed: {len(summary)}; errors: {errors}")
    print(f"Summary: {args.output_folder.resolve() / 'batch_summary.csv'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
