"""CLI entry point: train a toy model on selective copying.

Usage (from training/):
    uv run python run_train.py --arch mamba
    uv run python run_train.py --arch gpt

Creates outputs/<arch>/<UTCtimestamp>-<git_short>/ with best.pt, latest.pt,
config_snapshot.json and metrics.jsonl (outputs/ is gitignored), then updates
the committed pointer file runs.json so downstream export (Task 7) can find
checkpoints deterministically.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from llmviz_train.config import load_config
from llmviz_train.train import DEVICE, train_loop

TRAINING_DIR = Path(__file__).resolve().parent
RUNS_JSON = TRAINING_DIR / "runs.json"


def git_short() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=TRAINING_DIR,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def update_runs_json(arch: str, entry: dict) -> None:
    """Merge-update the committed pointer file, atomically."""
    runs = {}
    if RUNS_JSON.exists():
        with open(RUNS_JSON) as f:
            runs = json.load(f)
    runs[arch] = entry
    tmp = RUNS_JSON.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(runs, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, RUNS_JSON)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=["mamba", "gpt", "rwkv", "moe", "kan"])
    args = parser.parse_args()

    cfg = load_config()
    sha = git_short()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = TRAINING_DIR / "outputs" / args.arch / f"{timestamp}-{sha}"
    run_dir.mkdir(parents=True, exist_ok=False)

    snapshot = {
        "arch": args.arch,
        "config": cfg,
        "device": DEVICE,
        "git_short": sha,
        "started_utc": timestamp,
        "argv": sys.argv,
    }
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)
        f.write("\n")

    print(f"run dir: {run_dir}")
    result = train_loop(args.arch, cfg, out_dir=run_dir)

    target = cfg["train"]["val_acc_target"]
    if result["best_val_exact"] < target:
        print(
            f"FAILED: best val exact-match {result['best_val_exact']:.4f} < "
            f"target {target} after {result['final_step']} steps. "
            "Not updating runs.json.",
            flush=True,
        )
        raise SystemExit(1)

    finished = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    update_runs_json(
        args.arch,
        {
            "run_dir": str(run_dir.relative_to(TRAINING_DIR)),
            "best_val_exact": result["best_val_exact"],
            "step": result["best_step"],
            "git_short": sha,
            "finished_utc": finished,
        },
    )
    print(f"updated {RUNS_JSON.relative_to(TRAINING_DIR)}: {args.arch} -> "
          f"best_val_exact={result['best_val_exact']:.4f} @ step {result['best_step']}")


if __name__ == "__main__":
    main()
