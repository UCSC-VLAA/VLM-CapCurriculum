# `data_pipeline/` — Building Stage 1 perception data

This module constructs **`D_perc`**, the perception-difficulty dataset used in
Stage 1 of the staged training recipe (Section 3 of the paper).

## What this builds

Given an image-caption corpus (DOCCI or PixmoCap), we want training samples
that **specifically isolate perception failures**: an MCQ that the model gets
wrong when shown the image, but right when given the caption. Formally, for
each `(I, C, Q, A)`:

```
keep iff  Â_img(Q | I) ≠ A   ∧   Â_cap(Q | C) = A
```

The full pipeline is four steps:

```
   captions C                      ┌──────────────────────────────────┐
       │                           │      filter_perception.py        │
       ▼                           │                                  │
 generate_qa.py  ──►  raw MCQs ───►│  --mode image    (two VLMs)      │
   (Qwen2.5-72B,                   │  --mode caption  (two VLMs)      │
    one prompt per                 │  --mode filter   (intersect)     │
    image-caption                  │                                  │
    pair)                          └────────────────┬─────────────────┘
                                                    │
                                                    ▼
                                       filtered_perception.jsonl
                                                    │
                                                    ▼  format_for_training.py
                                       train/val/test_stage1_perception.jsonl
                                       (drop into EasyR1 as data.train_files)
```

We use two filter VLMs (default Qwen2.5-VL-7B and Qwen2.5-VL-32B) and keep
the **intersection** — only samples both models miss from the image *and*
both solve from the caption — to suppress filter-model artefacts.

## Inputs

`generate_qa.py` consumes a JSONL with one image-caption record per line:

| source | required field | example |
|---|---|---|
| `docci`  | `description`, `image_file` (and any extra meta you want to keep) | DOCCI's `docci_descriptions.jsonl` |
| `pixmo`  | `caption`, `image_url` | PixmoCap's caption JSONL |

The MCQ JSON returned by the LLM is parsed by a fenced-block regex; malformed
generations are skipped (logged to stderr).

## Outputs

`format_for_training.py` writes EasyR1-ready JSONL (matches the schema EasyR1
loads via `data.prompt_key=prompt`, `data.image_key=image_path`). Splits are
hash-deterministic (default 85% train / 5% val / 10% test, controllable).

## Quickstart

End-to-end (one MCQ source, two filter VLMs):

```bash
SOURCE=docci \
CAPTIONS=/path/to/docci_descriptions.jsonl \
IMAGE_ROOT=/path/to/DOCCI/images_downsampled_2x \
OUT_DIR=./outputs \
bash examples/run_full_pipeline.sh
```

The script will eventually print the EasyR1 launcher knobs to point at the
generated train/val files.

## Per-script CLI

| Script | What it does |
|---|---|
| `generate_qa.py` | vLLM-based caption → MCQ generation. Reads `prompts/{docci,pixmo}_mcq_generation.txt`. |
| `filter_perception.py` | Three modes (`image` / `caption` / `filter`). The first two run vLLM inference; the third is a pure JSON set-intersection. |
| `format_for_training.py` | filtered MCQs → `{train,val,test}_stage1_perception.jsonl` in EasyR1 schema. |

Each script accepts `--help` for its full flags.

## Reusing for other capability-curriculum experiments

The pipeline is intentionally factored so the filter step can be swapped:
to target a different perception subskill, change the **prompt** under
`prompts/`, or replace `filter_perception.py`'s decision rule (single-line
function `run_filter`). Stage 2 (textual reasoning) and Stage 3 (visual
reasoning) datasets are reused from existing public sources and are not
generated here — see `training/README.md`.

## Notes on heritage

- The generation step is a self-contained rewrite of
  `juncheng/export/SimpleInfer/{llm_inference_vllm.py,dataset/base.py}`.
- The filtering step formerly lived inside VLMEvalKit
  (`vlmeval/dataset/image_mcq.py`: `DOCCIDataset`, `DOCCIDataset_w_Caption`,
  `PixmoCapDataset`, `PixmoCapDataset_w_Caption`). It has been decoupled
  here to avoid the dependency.
