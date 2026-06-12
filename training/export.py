"""CLI wrapper: export trained weights + goldens for the TS site.

All logic lives in llmviz_train/export.py; this file is argparse + paths
only (mirrors the run_train.py split). Checkpoints are selected strictly
via runs.json. Site assets (manifest.json + weights.bin) go under
--models-root; goldens.json (test fixture, not a site asset) goes under
--goldens-root.

Usage (from training/):
    uv run python export.py                 # both archs
                                            #   -> <repo>/public/models/<arch>/
                                            #   -> <repo>/goldens/<arch>/
    uv run python export.py --arch gpt
    uv run python export.py --models-root /tmp/models --goldens-root /tmp/goldens
"""

import argparse
import json
from pathlib import Path

from llmviz_train.export import DEFAULT_GOLDENS_ROOT, DEFAULT_MODELS_ROOT, export_arch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["mamba", "gpt", "all"], default="all")
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--goldens-root", type=Path, default=DEFAULT_GOLDENS_ROOT)
    args = parser.parse_args()

    archs = ["mamba", "gpt"] if args.arch == "all" else [args.arch]
    for arch in archs:
        summary = export_arch(arch, args.models_root / arch, args.goldens_root / arch)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
