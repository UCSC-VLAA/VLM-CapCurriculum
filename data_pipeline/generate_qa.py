#!/usr/bin/env python3
"""Generate perception MCQs from image-caption datasets via vLLM.

Reads a caption corpus (DOCCI or PixmoCap), prompts an LLM (default
Qwen2.5-72B-Instruct) to author one 4-way MCQ per caption, and writes
``{generated, meta}`` rows to a JSONL file.

Example
-------
    python generate_qa.py \
        --captions /path/to/docci_descriptions.jsonl \
        --source docci \
        --model Qwen/Qwen2.5-72B-Instruct \
        --output outputs/DOCCI_Qwen25_72B.jsonl \
        --tensor-parallel-size 8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


PROMPT_DIR = Path(__file__).parent / "prompts"

CAPTION_KEY = {
    "docci": "description",
    "pixmo": "caption",
}

PROMPT_FILE = {
    "docci": "docci_mcq_generation.txt",
    "pixmo": "pixmo_mcq_generation.txt",
}


def load_captions(path: str, limit: int = -1) -> List[Dict[str, Any]]:
    """Load a JSONL caption file. Each line must contain the caption field
    expected for the corresponding source (see ``CAPTION_KEY``)."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if 0 < limit == len(records):
                break
    return records


def load_template(source: str) -> str:
    return (PROMPT_DIR / PROMPT_FILE[source]).read_text(encoding="utf-8")


def build_prompt(template: str, caption: str) -> str:
    return template.format(description=caption)


def parse_mcq(raw: str) -> Dict[str, Any]:
    """Extract the JSON MCQ from a fenced code block in the model output."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if not match:
        raise ValueError("No ```json fenced block found in generation.")
    return json.loads(match.group(1))


def batched(it: Iterable[Any], n: int) -> Iterator[List[Any]]:
    batch: List[Any] = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", required=True, help="JSONL of image-caption records")
    ap.add_argument("--source", required=True, choices=sorted(CAPTION_KEY.keys()))
    ap.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=-1, help="Process at most N records (debug)")
    ap.add_argument("--batch-size", type=int, default=8)

    # vLLM tuning
    ap.add_argument("--tensor-parallel-size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--swap-space", type=int, default=4)
    ap.add_argument("--trust-remote-code", action="store_true")

    # Sampling
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)

    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # imported lazily so --help works without vllm
    from tqdm import tqdm

    captions = load_captions(args.captions, limit=args.limit)
    template = load_template(args.source)
    cap_key = CAPTION_KEY[args.source]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        max_seq_len_to_capture=args.max_seq_len,
        trust_remote_code=args.trust_remote_code,
    )
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

    n_ok = n_skip = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for batch in tqdm(
            batched(captions, args.batch_size),
            desc="Generating MCQs",
            total=(len(captions) + args.batch_size - 1) // args.batch_size,
            file=sys.stdout,
        ):
            prompts = [build_prompt(template, rec.get(cap_key, "")) for rec in batch]
            outputs = llm.generate(prompts, sp)
            for rec, out in zip(batch, outputs):
                text = out.outputs[0].text
                try:
                    mcq = parse_mcq(text)
                except Exception as e:
                    n_skip += 1
                    print(f"[skip] {e}: {text[:120]!r}", file=sys.stderr)
                    continue
                fout.write(json.dumps({"generated": mcq, "meta": rec}, ensure_ascii=False) + "\n")
                n_ok += 1

    print(f"[✓] wrote {n_ok} MCQs → {out_path}  (skipped {n_skip})")


if __name__ == "__main__":
    main()
