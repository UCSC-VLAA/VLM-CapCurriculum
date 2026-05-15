#!/usr/bin/env python3
"""Perception-difficulty filtering for synthesized MCQs.

For every (image I, caption C, MCQ Q→A), we keep the sample iff
    image-only inference is wrong  AND  caption-only inference is right
i.e. ``Â_img ≠ A  ∧  Â_cap = A``.

Robustness: by default we run this filter with two different VLMs (e.g.
Qwen2.5-VL-7B and Qwen2.5-VL-32B) and keep only the **intersection** —
samples both models miss from the image but solve from the caption. This
yields the perception-difficulty subset used to train Stage 1.

This script is fully self-contained — it does **not** depend on
VLMEvalKit. It uses vLLM's chat/multimodal API directly.

Modes
-----
    --mode image    run image-only inference, write predictions
    --mode caption  run caption-only inference, write predictions
    --mode filter   take 1+ image preds and 1+ caption preds and keep
                    samples where image is wrong AND caption is right
                    for ALL provided runs (i.e. set intersection over
                    failure-from-image and success-from-caption).

Example end-to-end (one MCQ source, two VLMs):

    # 1) image-only inference with two models
    python filter_perception.py --mode image  --mcq mcqs.jsonl \\
        --model Qwen/Qwen2.5-VL-7B-Instruct  --image-root /data/DOCCI/images \\
        --output preds_img_7b.jsonl
    python filter_perception.py --mode image  --mcq mcqs.jsonl \\
        --model Qwen/Qwen2.5-VL-32B-Instruct --image-root /data/DOCCI/images \\
        --output preds_img_32b.jsonl

    # 2) caption-only inference with the same two models
    python filter_perception.py --mode caption --mcq mcqs.jsonl \\
        --model Qwen/Qwen2.5-VL-7B-Instruct  --output preds_cap_7b.jsonl
    python filter_perception.py --mode caption --mcq mcqs.jsonl \\
        --model Qwen/Qwen2.5-VL-32B-Instruct --output preds_cap_32b.jsonl

    # 3) intersect
    python filter_perception.py --mode filter \\
        --mcq mcqs.jsonl \\
        --image-preds preds_img_7b.jsonl preds_img_32b.jsonl \\
        --caption-preds preds_cap_7b.jsonl preds_cap_32b.jsonl \\
        --output filtered_perception.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def batched(it: Iterable[Any], n: int) -> Iterator[List[Any]]:
    batch: List[Any] = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def mcq_id(rec: Dict[str, Any]) -> str:
    """Stable identifier for a generated MCQ.

    Falls back through several common caption-source schemas
    (DOCCI: example_id, PixmoCap: image_url) before hashing the meta block.
    """
    meta = rec.get("meta", rec)
    for key in ("example_id", "image_url", "id"):
        if key in meta:
            return str(meta[key])
    return str(hash(json.dumps(meta, sort_keys=True)))


def build_question_block(mcq: Dict[str, Any]) -> str:
    options = mcq["options"]
    lines = [f"Question: {mcq['question']}", "Options:"]
    for letter in ("A", "B", "C", "D"):
        lines.append(f"{letter}. {options[letter]}")
    lines.append("Please select the correct answer from the options above.")
    return "\n".join(lines)


_LETTER_RE = re.compile(r"\b([ABCD])\b")


def extract_letter(text: str) -> str:
    """Best-effort A/B/C/D extraction from a model response."""
    text = text.strip()
    # Common pattern: "Answer: A" or "The answer is B."
    m = re.search(r"answer\s*[:\-]?\s*([ABCD])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = _LETTER_RE.search(text.upper())
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Inference modes
# ---------------------------------------------------------------------------

def _resolve_image_path(image_field: Any, image_root: Path) -> str:
    """meta["image_file"] / meta["image_url"] / meta["image_path"] -> path on disk."""
    if isinstance(image_field, list):
        image_field = image_field[0]
    if isinstance(image_field, str) and image_field.startswith(("http://", "https://")):
        return image_field
    return str(image_root / str(image_field))


def run_image_only(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams
    from PIL import Image
    from tqdm import tqdm

    rows = read_jsonl(args.mcq)
    image_root = Path(args.image_root) if args.image_root else None

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    convs: List[Dict[str, Any]] = []
    for rec in rows:
        meta = rec.get("meta", {})
        image_field = (
            meta.get("image_file")
            or meta.get("image_path")
            or meta.get("image_url")
        )
        if image_field is None:
            raise KeyError(f"no image field in meta for record id={mcq_id(rec)}")
        img_path = _resolve_image_path(image_field, image_root) if image_root else image_field
        try:
            image = Image.open(img_path).convert("RGB") if not str(img_path).startswith("http") else img_path
        except Exception as e:
            print(f"[skip] cannot load {img_path}: {e}", file=sys.stderr)
            convs.append(None)
            continue
        question_block = build_question_block(rec["generated"])
        convs.append({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Please answer based on the image.\n" + question_block},
                ],
            }],
        })

    out_rows: List[Dict[str, Any]] = []
    for batch_idx, batch in enumerate(tqdm(list(batched(zip(rows, convs), args.batch_size)),
                                           desc="image-only inference")):
        live = [(r, c) for r, c in batch if c is not None]
        if not live:
            continue
        outs = llm.chat([c["messages"] for _, c in live], sp)
        for (rec, _), out in zip(live, outs):
            text = out.outputs[0].text
            out_rows.append({
                "id": mcq_id(rec),
                "ground_truth": rec["generated"]["answer"],
                "predicted_letter": extract_letter(text),
                "raw": text,
            })

    write_jsonl(args.output, out_rows)
    print(f"[✓] image-only predictions → {args.output}  ({len(out_rows)}/{len(rows)})")


def run_caption_only(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams
    from tqdm import tqdm

    rows = read_jsonl(args.mcq)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    prompts: List[str] = []
    for rec in rows:
        meta = rec.get("meta", {})
        caption = meta.get("description") or meta.get("caption") or ""
        question_block = build_question_block(rec["generated"])
        prompts.append(
            "Please answer the question based only on the image caption below.\n"
            f"Image Caption: {caption}\n\n{question_block}"
        )

    out_rows: List[Dict[str, Any]] = []
    for batch in tqdm(list(batched(zip(rows, prompts), args.batch_size)),
                      desc="caption-only inference"):
        outs = llm.generate([p for _, p in batch], sp)
        for (rec, _), out in zip(batch, outs):
            text = out.outputs[0].text
            out_rows.append({
                "id": mcq_id(rec),
                "ground_truth": rec["generated"]["answer"],
                "predicted_letter": extract_letter(text),
                "raw": text,
            })

    write_jsonl(args.output, out_rows)
    print(f"[✓] caption-only predictions → {args.output}  ({len(out_rows)}/{len(rows)})")


# ---------------------------------------------------------------------------
# Filter mode (no model needed)
# ---------------------------------------------------------------------------

def _index_by_id(preds_path: str) -> Dict[str, Dict[str, Any]]:
    return {row["id"]: row for row in read_jsonl(preds_path)}


def run_filter(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.mcq)
    image_pred_sets = [_index_by_id(p) for p in args.image_preds]
    caption_pred_sets = [_index_by_id(p) for p in args.caption_preds]

    n_total = len(rows)
    n_kept = 0
    out_rows: List[Dict[str, Any]] = []
    for rec in rows:
        rid = mcq_id(rec)
        gt = rec["generated"]["answer"]

        # ALL image runs must be wrong
        all_image_wrong = all(
            (preds.get(rid, {}).get("predicted_letter") not in ("", gt))
            for preds in image_pred_sets
        )
        if not all_image_wrong:
            continue

        # ALL caption runs must be right
        all_caption_right = all(
            (preds.get(rid, {}).get("predicted_letter") == gt)
            for preds in caption_pred_sets
        )
        if not all_caption_right:
            continue

        out_rows.append(rec)
        n_kept += 1

    write_jsonl(args.output, out_rows)
    keep_pct = 100.0 * n_kept / max(n_total, 1)
    print(f"[✓] filtered: kept {n_kept}/{n_total} ({keep_pct:.1f}%) → {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["image", "caption", "filter"])
    ap.add_argument("--mcq", required=True, help="JSONL of generated MCQs (output of generate_qa.py)")
    ap.add_argument("--output", required=True)

    # inference modes
    ap.add_argument("--model", help="HF id or local path (image / caption modes)")
    ap.add_argument("--image-root", default=None, help="Directory prepended to meta.image_file / image_path")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tensor-parallel-size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=8)

    # filter mode
    ap.add_argument("--image-preds", nargs="+", default=[],
                    help="One or more *.jsonl files from --mode image runs")
    ap.add_argument("--caption-preds", nargs="+", default=[],
                    help="One or more *.jsonl files from --mode caption runs")

    args = ap.parse_args()

    if args.mode == "image":
        if not args.model:
            ap.error("--model is required for --mode image")
        run_image_only(args)
    elif args.mode == "caption":
        if not args.model:
            ap.error("--model is required for --mode caption")
        run_caption_only(args)
    else:
        if not args.image_preds or not args.caption_preds:
            ap.error("--image-preds and --caption-preds are required for --mode filter")
        run_filter(args)


if __name__ == "__main__":
    main()
