<div align="center">

# VLM-CapCurriculum

### From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models

*See first, then think — and treat **capability** as a new curriculum axis.*

[![Paper](https://img.shields.io/badge/Paper-ICML%202026-b31b1b)](#citation)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![HF Collection](https://img.shields.io/badge/🤗%20HF-Collection-yellow)](#resources)
[![Website](https://img.shields.io/badge/Project-Page-green)](#resources)

</div>

---

## TL;DR

**Visual perception, not reasoning length, is the dominant bottleneck for visual reasoning in VLMs.** We show that **86.9%** of incorrect answers from a state-of-the-art Qwen3-VL-8B trace back to perception errors that no amount of additional thinking can fix — *longer thinking cannot rescue a wrong "look."*

So we **see first, then think.** Concretely, we decouple post-training into three stages along a capability axis:

```
   Stage 1                Stage 2               Stage 3
 Visual Perception  →   Textual Reasoning  →  Visual Reasoning
   (D_perc, RLVR)        (D_text, RLVR)        (D_vis, RLVR)
```

Across **four backbones** (Qwen2.5-VL-7B, Qwen3-VL-8B, InternVL3-8B, InternVL3.5-8B) this *staged* recipe consistently beats the standard *merged* baseline that pools all data into one stage. On Qwen3-VL-8B it yields **+1.46% accuracy** with **20.8% shorter** reasoning traces — better perception literally lets the model think less.

> 🧭 **Conceptually**, this staging is best understood as a **new curriculum axis: the capability axis** (perception → reasoning), orthogonal to the classic *difficulty axis* (easy → hard). They **stack additively** — combining both lifts Qwen3-VL-8B from 58.6 → **63.0** average, beating either axis alone by >2 points. See [§ A new curriculum dimension](#a-new-curriculum-dimension) below.

---

## A new curriculum dimension

Curriculum learning has historically meant **ordering training samples by difficulty** (easy → hard). Our staged recipe surfaces a second, orthogonal axis that has been largely overlooked in VLM post-training: **what capability each batch trains**. Sample difficulty *and* the capability under training are two independent knobs:

```
                Capability axis (ours)
                  perception → reasoning
                       ▲
                       │
   ┌───────────────────┼───────────────────┐
   │   Capability ✓    │  Capability ✓     │
   │   Difficulty ✗    │  Difficulty ✓     │   ← additive sweet spot
   ├───────────────────┼───────────────────┤
   │   Capability ✗    │  Capability ✗     │
   │   Difficulty ✗    │  Difficulty ✓     │   ← prior curriculum work
   └───────────────────┴───────────────────► Difficulty axis
                  easy → hard
```

Empirically the two axes **stack additively** on Qwen3-VL-8B:

| Curriculum | Avg over 7 benchmarks | Δ over Merged |
|---|:---:|:---:|
| None (Merged baseline) | 58.56 | — |
| Difficulty only | 60.36 | +1.80 |
| **Capability only (ours)** | **60.53** | **+1.97** |
| **Capability + Difficulty** | **62.99** | **+4.43** |

This reframes post-training as choosing a *trajectory through a 2D space* rather than along a single difficulty line, and opens a search space (other capability decompositions, joint schedules, etc.) that prior curriculum work has not touched. See Section 4.5 of the paper for details and [`training/examples/curriculum/`](training/examples/curriculum/) for the launch scripts.

---

## Headline numbers

### Qwen3-VL-8B base vs. our staged recipe

| Setting | Visual Math AVG | Perception AVG | Overall AVG |
|---|:---:|:---:|:---:|
| Qwen3-VL-8B base | 45.17 | 79.21 | 62.19 |
| Qwen3-VL-8B + Merged training | 49.64 | 79.71 | 64.67 |
| **Qwen3-VL-8B + Staged (ours)** | **51.10** | **80.44** | **65.77** |
| OneThinker-8B (concurrent baseline) | 51.10 | 78.64 | 64.87 |

Visual math = MathVista / MathVision / MathVerse(VI) / WeMath. Perception = A-OKVQA / RealWorldQA / MMStar / POPE.

### Staged > Merged across four backbones

| Backbone | Δ Overall AVG (Staged − Merged) |
|---|:---:|
| Qwen3-VL-8B | **+3.37** |
| InternVL3-8B | **+3.77** |
| Qwen2.5-VL-7B | +1.62 |
| InternVL3.5-8B | +0.95 |

(see `docs/results.md` for the full extended-benchmark table from the paper appendix.)

*(For the additive Capability × Difficulty result, see [§ A new curriculum dimension](#a-new-curriculum-dimension) above.)*

---

## Repository layout

```
VLM-CapCurriculum/
├── data_pipeline/      # Stage 1 perception data synthesis (DOCCI/Pixmo → MCQ → filter)
├── training/           # GRPO/RLVR training scripts on top of EasyR1
│   ├── examples/       # one .sh per stage, per backbone
│   ├── reward_functions/
│   └── format_prompts/
├── evaluation/         # VLMEvalKit configs + Claude-Haiku-4.5 judge setup
│   └── perception_error_analysis/   # Sec 4.4 analysis pipeline
├── scripts/
│   └── quickstart_qwen3vl_8b_staged.sh    # one-shot stage1→2→3
├── docs/
│   ├── images/         # paper figures (teaser, pipeline, case study)
│   └── results.md      # full benchmark tables
└── requirements.txt
```

The repo is **paper-specific** — it does not vendor [EasyR1](https://github.com/hiyouga/EasyR1) or [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Install them separately and point our scripts at them. See [Setup](#setup).

---

## Resources

- 🤗 **Collection** (single hub for everything below): [`UCSC-VLAA / VLM-CapCurriculum`](https://huggingface.co/collections/UCSC-VLAA/vlm-capcurriculum-from-seeing-to-thinking-icml-2026-6a07691f944148ccb2b183b8)
- 🤗 **Models**:
  - [`UCSC-VLAA/VLM-CapCurriculum-Qwen3-VL-8B-Staged`](https://huggingface.co/UCSC-VLAA/VLM-CapCurriculum-Qwen3-VL-8B-Staged) *(primary, ICML headline numbers)*
  - [`UCSC-VLAA/VLM-CapCurriculum-Qwen2.5-VL-7B-Staged`](https://huggingface.co/UCSC-VLAA/VLM-CapCurriculum-Qwen2.5-VL-7B-Staged)
  - [`UCSC-VLAA/VLM-CapCurriculum-InternVL3-8B-Staged`](https://huggingface.co/UCSC-VLAA/VLM-CapCurriculum-InternVL3-8B-Staged) *(largest staged-vs-merged delta)*
  - [`UCSC-VLAA/VLM-CapCurriculum-InternVL3.5-8B-Staged`](https://huggingface.co/UCSC-VLAA/VLM-CapCurriculum-InternVL3.5-8B-Staged)
- 🤗 **Datasets** (each ships with `pass_rate` for difficulty curricula):
  - [`UCSC-VLAA/VLM-CapCurriculum-Perception-Data`](https://huggingface.co/datasets/UCSC-VLAA/VLM-CapCurriculum-Perception-Data) — Stage 1 (synthesised + filtered DOCCI MCQs)
  - [`UCSC-VLAA/VLM-CapCurriculum-TextReasoning-Data`](https://huggingface.co/datasets/UCSC-VLAA/VLM-CapCurriculum-TextReasoning-Data) — Stage 2 (ORZ-Math-13k)
  - [`UCSC-VLAA/VLM-CapCurriculum-VisualReasoning-Data`](https://huggingface.co/datasets/UCSC-VLAA/VLM-CapCurriculum-VisualReasoning-Data) — Stage 3 (CLEVR-Math + GeoQA170K + Math PUMA + ArxivQA)
- 🌐 **Project page**: [ucsc-vlaa.github.io/VLM-CapCurriculum](https://ucsc-vlaa.github.io/VLM-CapCurriculum) *(coming soon — see Roadmap below)*

---

## Roadmap

A few items are intentionally still in flight at first release. We track them here so contributors and readers know what to expect:

- [ ] **Paper URL** — replace the OpenReview/arXiv placeholder once the camera-ready link is live. Touches: `CITATION.cff`, all four HF model cards, all three HF dataset cards, the project page hero.
- [ ] **Author list** — replace the `TBD` author entries with the final author names + affiliations once de-anonymized. Touches: `CITATION.cff`, all seven HF cards, the project page hero (`docs/index.html`).
- [ ] **Project website polish** — the static site under [`docs/`](./docs) is live at [`ucsc-vlaa.github.io/VLM-CapCurriculum`](https://ucsc-vlaa.github.io/VLM-CapCurriculum). Open placeholders to fill once the paper is ready:
  - "Author list TBD" line in the hero (`docs/index.html`)
  - the **Paper** button in the hero — currently links to `#`
  - BibTeX block in the *Cite* section — `author = {TBD}`
  - longer-term: replace the static result tables with rendered numbers pulled from `eval-results/` once that dataset exists.

If any of these block your reproduction, please open an issue.

---

## Setup

```bash
git clone https://github.com/UCSC-VLAA/VLM-CapCurriculum.git
cd VLM-CapCurriculum

# 1) Paper-side deps (data synthesis + Claude judge)
pip install -r requirements.txt

# 2) Training framework — install separately
git clone https://github.com/hiyouga/EasyR1.git ../EasyR1
cd ../EasyR1 && pip install -e . && cd -

# 3) Evaluation framework — install separately
git clone https://github.com/open-compass/VLMEvalKit.git ../VLMEvalKit
cd ../VLMEvalKit && pip install -e . && cd -
```

Set environment paths used by our scripts:

```bash
export EASYR1_HOME=$(realpath ../EasyR1)
export VLMEVALKIT_HOME=$(realpath ../VLMEvalKit)
export VLMCC_HOME=$(pwd)
```

---

## Quickstart — Reproduce Qwen3-VL-8B (Staged)

The end-to-end recipe takes ~24 GPU-hours on 8× H200.

```bash
# (a) Build perception data D_perc (or download from HF; see data_pipeline/README.md)
bash data_pipeline/examples/run_full_pipeline.sh

# (b) Run Stage 1 → Stage 2 → Stage 3 training in one shot
bash scripts/quickstart_qwen3vl_8b_staged.sh

# (c) Evaluate on the 8 benchmarks reported in the paper
bash evaluation/run_eval.sh <CHECKPOINT_DIR>
```

For more granular runs (per-stage, ablations, other backbones) see `training/examples/`.

---

## Detailed reproduction guides

| Topic | Pointer |
|---|---|
| Synthesizing & filtering Stage 1 perception data | [`data_pipeline/README.md`](data_pipeline/README.md) |
| Per-stage training scripts | [`training/examples/`](training/examples/) |
| Reward functions & prompt templates | [`training/reward_functions/`](training/reward_functions/), [`training/format_prompts/`](training/format_prompts/) |
| VLMEvalKit configs & judge setup | [`evaluation/README.md`](evaluation/README.md) |
| Sec 4.4 perception-error analysis | [`evaluation/perception_error_analysis/`](evaluation/perception_error_analysis/) |
| Ablations (stage order, encoder freezing, SFT vs RL) | [`training/examples/ablations/`](training/examples/ablations/) |
| Difficulty + Capability curriculum (Sec 4.5) | [`training/examples/curriculum/`](training/examples/curriculum/) |

---

## Citation

```bibtex
@inproceedings{vlmcapcurriculum2026,
  title  = {From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models},
  author = {TODO},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year   = {2026}
}
```

---

## Acknowledgements

This work builds on top of the open-source releases of
[EasyR1](https://github.com/hiyouga/EasyR1),
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit),
[Qwen2.5-VL](https://huggingface.co/Qwen) / [Qwen3-VL](https://huggingface.co/Qwen),
[InternVL3](https://huggingface.co/OpenGVLab),
[DOCCI](https://google.github.io/docci/), and
[PixmoCap](https://huggingface.co/datasets/allenai/pixmo-cap).
We thank the maintainers of these projects.
