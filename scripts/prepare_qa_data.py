#!/usr/bin/env python3
"""Download the CORE evaluation bundle used by `--task core` (QA evaluation).

The bundle ships ``core.yaml``, ``eval_meta_data.csv`` and ``eval_data/`` -- the
QA benchmark data for every CORE task. It is the same artifact nanochat fetches
and unzips, so this is the only download needed before running `runs/qa.sh`.

Usage:
    python -m scripts.prepare_qa_data                 # -> <repo>/eval_bundle
    python -m scripts.prepare_qa_data --dest /path    # custom location
    python -m scripts.prepare_qa_data --verify-only
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
DEFAULT_DEST = REPO_ROOT / "eval_bundle"
REQUIRED = ("core.yaml", "eval_meta_data.csv", "eval_data")

# CORE tasks this project reports (subset of the bundle's task list).
BENCHMARKS = [
    ("agi_eval_lsat_ar", "AGI-Eval LSAT-AR"),
    ("arc_challenge", "ARC-Challenge"),
    ("arc_easy", "ARC-Easy"),
    ("bigbench_cs_algorithms", "BIG-bench CS Algorithms"),
    ("bigbench_dyck_languages", "BIG-bench Dyck Languages"),
    ("bigbench_language_identification", "BIG-bench Language Identification"),
    ("bigbench_operators", "BIG-bench Operators"),
    ("bigbench_qa_wikidata", "BIG-bench QA-Wikidata"),
    ("boolq", "BoolQ"),
    ("commonsense_qa", "CommonsenseQA"),
    ("copa", "COPA"),
    ("coqa", "CoQA"),
    ("hellaswag", "HellaSwag"),
    ("hellaswag_zeroshot", "HellaSwag-ZeroShot"),
    ("lambada_openai", "LAMBADA-OpenAI"),
    ("openbook_qa", "OpenBookQA"),
    ("piqa", "PIQA"),
    ("winograd", "Winograd"),
    ("winogrande", "WinoGrande"),
]


def is_complete(dest: Path) -> bool:
    return all((dest / name).exists() for name in REQUIRED)


def download(dest: Path) -> None:
    import requests

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "eval_bundle.zip"
        print(f"downloading {EVAL_BUNDLE_URL}")
        response = requests.get(EVAL_BUNDLE_URL, stream=True, timeout=60)
        response.raise_for_status()
        with open(zip_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)
        extracted = Path(tmp) / "eval_bundle"
        if not extracted.is_dir():
            raise RuntimeError("eval_bundle directory not found inside the archive")
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(dest))
    print(f"placed eval bundle at {dest}")


def verify(dest: Path) -> int:
    import yaml

    missing_files = [name for name in REQUIRED if not (dest / name).exists()]
    if missing_files:
        print(f"ERROR: incomplete bundle at {dest}; missing {missing_files}")
        return 1

    labels = {task["label"] for task in yaml.safe_load((dest / "core.yaml").read_text())["icl_tasks"]}
    absent = [label for label, _ in BENCHMARKS if label not in labels]
    print(f"core.yaml tasks: {len(labels)}")
    print(f"BENCHMARKS covered: {len(BENCHMARKS) - len(absent)}/{len(BENCHMARKS)}")
    if absent:
        print(f"  missing from bundle: {absent}")
        return 1
    print(f"  extra tasks in bundle: {sorted(labels - {label for label, _ in BENCHMARKS})}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    if not args.verify_only:
        if is_complete(args.dest) and not args.force:
            print(f"{args.dest} already complete; skipping download (use --force to refresh)")
        else:
            download(args.dest)
    return verify(args.dest)


if __name__ == "__main__":
    sys.exit(main())
