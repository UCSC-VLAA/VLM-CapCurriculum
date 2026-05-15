#!/usr/bin/env python3
"""Apply the BedrockClaude judge patch to a local VLMEvalKit checkout.

Three edits are made:

  1. Copy `bedrock_claude.py` into ``<VLMEVALKIT>/vlmeval/api/``
  2. Insert ``from .bedrock_claude import BedrockClaude`` and the
     ``BedrockClaude`` entry into ``<VLMEVALKIT>/vlmeval/api/__init__.py``
  3. Register the ``bedrock-claude-haiku-4.5`` /
     ``bedrock-claude-sonnet-4.5`` aliases in
     ``<VLMEVALKIT>/vlmeval/dataset/utils/judge_util.py``

All edits are idempotent — re-running this script is safe.

Usage:
    python evaluation/vlmevalkit_patches/apply_patches.py /path/to/VLMEvalKit
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


PATCH_DIR = Path(__file__).resolve().parent
SRC_BEDROCK = PATCH_DIR / "api" / "bedrock_claude.py"

API_INIT_IMPORT = "from .bedrock_claude import BedrockClaude"
API_INIT_ALL = "'BedrockClaude'"

JUDGE_UTIL_IMPORT_OLD = "from ...api import OpenAIWrapper, SiliconFlowAPI, HFChatModel"
JUDGE_UTIL_IMPORT_NEW = "from ...api import OpenAIWrapper, SiliconFlowAPI, HFChatModel, BedrockClaude"

JUDGE_UTIL_ALIASES = (
    "            # Claude 4.5 models (added by VLM-CapCurriculum)\n"
    "            'bedrock-claude-haiku-4.5':  'global.anthropic.claude-haiku-4-5-20251001-v1:0',\n"
    "            'bedrock-claude-sonnet-4.5': 'global.anthropic.claude-sonnet-4-5-20250929-v1:0',\n"
)

JUDGE_UTIL_DISPATCH = (
    "    if model and model.startswith('bedrock-claude'):\n"
    "        model = BedrockClaude(model=model_version, **kwargs)\n"
    "    elif "
)


def step_copy_file(vlmevalkit_root: Path) -> None:
    dst = vlmevalkit_root / "vlmeval" / "api" / "bedrock_claude.py"
    if dst.exists() and dst.read_bytes() == SRC_BEDROCK.read_bytes():
        print(f"  [skip] {dst.relative_to(vlmevalkit_root)} already up to date")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_BEDROCK, dst)
    print(f"  [ok]   copied {dst.relative_to(vlmevalkit_root)}")


def step_patch_api_init(vlmevalkit_root: Path) -> None:
    path = vlmevalkit_root / "vlmeval" / "api" / "__init__.py"
    src = path.read_text(encoding="utf-8")
    changed = False

    if API_INIT_IMPORT not in src:
        # Insert the import after the last `from .` import line in the file.
        lines = src.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("from .") or line.startswith("import "):
                insert_at = i + 1
        lines.insert(insert_at, API_INIT_IMPORT + "\n")
        src = "".join(lines)
        changed = True
        print(f"  [ok]   inserted import in {path.relative_to(vlmevalkit_root)}")
    else:
        print(f"  [skip] import already present in {path.relative_to(vlmevalkit_root)}")

    if "'BedrockClaude'" not in src and '"BedrockClaude"' not in src:
        # Append into __all__ if present; otherwise add a new __all__.
        m = re.search(r"__all__\s*=\s*\[([^\]]*)\]", src, re.DOTALL)
        if m:
            inner = m.group(1).rstrip()
            if inner and not inner.rstrip().endswith(","):
                inner = inner + ","
            new_block = f"__all__ = [{inner}\n    {API_INIT_ALL},\n]"
            src = src[: m.start()] + new_block + src[m.end():]
            changed = True
            print(f"  [ok]   added BedrockClaude to __all__")
        else:
            src += f"\n__all__ = [{API_INIT_ALL}]\n"
            changed = True
            print(f"  [ok]   created __all__ with BedrockClaude")
    else:
        print(f"  [skip] BedrockClaude already in __all__")

    if changed:
        path.write_text(src, encoding="utf-8")


def step_patch_judge_util(vlmevalkit_root: Path) -> None:
    path = vlmevalkit_root / "vlmeval" / "dataset" / "utils" / "judge_util.py"
    src = path.read_text(encoding="utf-8")
    changed = False

    # 1) extend the API import line
    if "BedrockClaude" not in src:
        if JUDGE_UTIL_IMPORT_OLD in src:
            src = src.replace(JUDGE_UTIL_IMPORT_OLD, JUDGE_UTIL_IMPORT_NEW, 1)
            changed = True
            print(f"  [ok]   extended API import in {path.relative_to(vlmevalkit_root)}")
        else:
            print("  [warn] could not find expected import line; please patch judge_util.py manually")
    else:
        print(f"  [skip] BedrockClaude import already present in judge_util.py")

    # 2) register the two aliases inside the existing model_map dict
    if "'bedrock-claude-haiku-4.5'" not in src:
        m = re.search(r"(model_map\s*=\s*\{)", src)
        if m:
            insert_pos = m.end()
            src = src[:insert_pos] + "\n" + JUDGE_UTIL_ALIASES + src[insert_pos:]
            changed = True
            print(f"  [ok]   registered bedrock-claude aliases in model_map")
        else:
            print("  [warn] could not locate model_map; please add aliases manually")
    else:
        print(f"  [skip] bedrock-claude aliases already in model_map")

    # 3) Insert the dispatch branch as the FIRST branch of the if/elif chain.
    #    We anchor on the first dispatch line ("if|elif model in ['qwen-7b'..."),
    #    rewrite it to start with our new branch, and demote the original `if`
    #    (if any) to an `elif` so the chain stays well-formed.
    if "BedrockClaude(model=model_version" not in src:
        anchor = re.search(
            r"^(?P<ws>[ \t]*)(?P<kw>if|elif)\s+model\s+in\s+\[\s*'qwen-7b'",
            src,
            re.MULTILINE,
        )
        if anchor:
            ws = anchor.group("ws")
            new_block = (
                f"{ws}if model and model.startswith('bedrock-claude'):\n"
                f"{ws}    model = BedrockClaude(model=model_version, **kwargs)\n"
                f"{ws}elif "
            )
            # Preserve everything after the original keyword.
            tail_start = anchor.end("kw")
            src = src[: anchor.start()] + new_block + src[tail_start:].lstrip(" \t")
            changed = True
            print(f"  [ok]   inserted bedrock-claude dispatch branch")
        else:
            print("  [warn] could not locate dispatch site; please add the branch manually")
    else:
        print(f"  [skip] bedrock-claude dispatch already present")

    if changed:
        path.write_text(src, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vlmevalkit", type=Path,
                    help="Path to your local VLMEvalKit clone "
                         "(the directory that contains a vlmeval/ subdir)")
    args = ap.parse_args()

    root = args.vlmevalkit.expanduser().resolve()
    if not (root / "vlmeval" / "api" / "__init__.py").exists():
        print(f"error: {root} does not look like a VLMEvalKit checkout", file=sys.stderr)
        return 2

    print(f"Patching {root}")
    print()
    print("(1/3) Copying api/bedrock_claude.py")
    step_copy_file(root)
    print()
    print("(2/3) Patching vlmeval/api/__init__.py")
    step_patch_api_init(root)
    print()
    print("(3/3) Patching vlmeval/dataset/utils/judge_util.py")
    step_patch_judge_util(root)
    print()
    print("Done. Verify with:")
    print(f"    python -c 'from vlmeval.api import BedrockClaude; print(BedrockClaude)'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
