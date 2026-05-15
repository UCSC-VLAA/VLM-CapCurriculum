#!/usr/bin/env python3
"""Format filtered perception MCQs into EasyR1 training JSONL.

Input  (one row per MCQ; output of ``filter_perception.py --mode filter``)::

    {
      "generated": {"question": "...", "options": {"A":..., "B":..., "C":..., "D":...}, "answer": "C"},
      "meta": {"example_id": "train_00009", "image_file": "train_00009.jpg", "description": "..."}
    }

Output (one row per MCQ; matches what EasyR1 expects via
``data.prompt_key=prompt`` and ``data.image_key=image_path``)::

    {
      "index": "0",
      "answer": "A",
      "example_id": "train_00009",
      "split": "train",
      "image_path": ["train_00009.jpg"],
      "option_A": "...",
      "option_B": "...",
      "option_C": "...",
      "option_D": "...",
      "prompt": "<image>In the image, ...\\nOptions:\\nA: ...\\nB: ...\\n...\\nRespond using only the letter corresponding to the correct answer.\\n"
    }

Splits are produced deterministically (hash-based) so re-runs are stable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROMPT_TEMPLATE = (
    "<image>{question}\n"
    "Options:\n"
    "A: {A}\n"
    "B: {B}\n"
    "C: {C}\n"
    "D: {D}\n"
    "Respond using only the letter corresponding to the correct answer.\n"
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def split_for(example_id: str, val_ratio: float, test_ratio: float) -> str:
    """Stable train/val/test assignment based on a hash of the example id."""
    h = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    if bucket < val_ratio:
        return "val"
    if bucket < val_ratio + test_ratio:
        return "test"
    return "train"


def to_training_row(rec: Dict[str, Any], index: int, split: str) -> Dict[str, Any]:
    mcq = rec["generated"]
    meta = rec.get("meta", {})

    image_field = (
        meta.get("image_file")
        or meta.get("image_path")
        or meta.get("image_url")
        or ""
    )
    image_paths = image_field if isinstance(image_field, list) else [image_field]

    example_id = (
        meta.get("example_id")
        or meta.get("id")
        or meta.get("image_url")
        or f"sample_{index:08d}"
    )

    prompt = PROMPT_TEMPLATE.format(
        question=mcq["question"].strip(),
        A=mcq["options"]["A"],
        B=mcq["options"]["B"],
        C=mcq["options"]["C"],
        D=mcq["options"]["D"],
    )

    return {
        "index": str(index),
        "answer": mcq["answer"],
        "example_id": example_id,
        "split": split,
        "image_path": image_paths,
        "option_A": mcq["options"]["A"],
        "option_B": mcq["options"]["B"],
        "option_C": mcq["options"]["C"],
        "option_D": mcq["options"]["D"],
        "prompt": prompt,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="JSONL of filtered MCQs")
    ap.add_argument("--output-dir", required=True,
                    help="Directory; writes train_/val_/test_<prefix>.jsonl")
    ap.add_argument("--prefix", default="stage1_perception",
                    help="Filename core, e.g. 'stage1_perception' → train_stage1_perception.jsonl")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    out_dir = Path(args.output_dir)

    buckets: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for i, rec in enumerate(rows):
        meta = rec.get("meta", {})
        eid = (
            meta.get("example_id")
            or meta.get("id")
            or meta.get("image_url")
            or f"sample_{i:08d}"
        )
        split = split_for(eid, args.val_ratio, args.test_ratio)
        buckets[split].append(to_training_row(rec, i, split))

    sizes = {}
    for split, rs in buckets.items():
        n = write_jsonl(out_dir / f"{split}_{args.prefix}.jsonl", rs)
        sizes[split] = n

    print(f"[✓] wrote → {out_dir}/")
    for split in ("train", "val", "test"):
        print(f"     {split:5s}: {sizes[split]}")


if __name__ == "__main__":
    main()
