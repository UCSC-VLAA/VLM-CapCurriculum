# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

# from mathruler.grader import extract_boxed_content, grade_answer

import multiprocessing as mp
import time
from mathruler.grader import extract_boxed_content, grade_answer

def _grade_worker(ans, gt, q):
    try:
        q.put(bool(grade_answer(ans, gt)))
    except Exception:
        q.put(False)

def grade_answer_timeout(ans: str, gt: str, timeout_s: float = 1.0) -> bool:
    q = mp.Queue(1)
    p = mp.Process(target=_grade_worker, args=(ans, gt, q), daemon=True)
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.kill()
        p.join()
        return False  # 超时当作不对（或你想给 0 分）
    return q.get() if not q.empty() else False

def too_complex(expr: str) -> bool:
    if len(expr) > 200:  # 可调
        return True
    if expr.count("^") + expr.count("**") > 10:
        return True
    if expr.count("(") > 40 or expr.count("{") > 40:
        return True
    # 超大整数
    import re
    if re.search(r"\d{50,}", expr):
        return True
    return False

def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    if not answer:
        return 0.0
    ok = False if too_complex(answer) else grade_answer(answer, ground_truth)
    return 1.0 if ok else 0.0



def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


# def accuracy_reward(response: str, ground_truth: str) -> float:
#     answer = extract_boxed_content(response)
#     return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
