#!/usr/bin/env python3
"""direct_answer_gpt54_cot.py

One-off sanity check: run gpt-5.4 direct-answer on v3/66 and v3/83 with
a prompt that forces visible <thinking> CoT (at least 100 words,
quoting from the story). The default direct-answer prompt let gpt-5.4
keep all reasoning in hidden reasoning tokens — this version demands
visible analysis so we can compare its reasoning to the free_choice
mode's reasoning on the same questions.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL = "gpt-5.4"
TEMPERATURE = 0.8
N_SAMPLES = 8
MAX_N_PER_CALL = 8
MAX_COMPLETION_TOKENS = 6000

TARGETS = [("v3", 66), ("v3", 83)]

SYSTEM = (
    "You are answering a reading-comprehension multiple-choice question. "
    "You will see the story, the question, and two candidate answers. "
    "Read carefully and pick the answer you believe is most supported "
    "by the story.\n\n"
    "REQUIRED: Before outputting your final answer, write at least 150 "
    "words of textual analysis inside <thinking></thinking> tags. The "
    "analysis must cite specific passages (using <quote>...</quote>) "
    "and reason about how each candidate answer is or isn't supported "
    "by the text. Do NOT skip the thinking step. Do NOT emit <answer> "
    "without at least 150 words of visible thinking first.\n\n"
    "After the thinking block, output your final answer as exactly "
    "<answer>A</answer> or <answer>B</answer>."
)

ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)
THINK_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)


def build_messages(question, answer_a, answer_b, story):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"<story>\n{story}\n</story>\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER A: {answer_a}\n"
            f"ANSWER B: {answer_b}\n\n"
            "First write at least 150 words of analysis with quotes "
            "inside <thinking></thinking>, then output "
            "<answer>A</answer> or <answer>B</answer>."
        )},
    ]


def parse_answer(text):
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    tail = text[-300:].upper()
    tm = re.search(r"\b([AB])\b", tail)
    return tm.group(1) if tm else None


async def call_once(messages, n):
    resp = await openai.ChatCompletion.acreate(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        n=n,
    )
    return [c["message"]["content"] for c in resp["choices"]]


async def call_gpt54(messages, n):
    chunks = []
    remaining = n
    while remaining > 0:
        k = min(MAX_N_PER_CALL, remaining)
        chunks.append(call_once(messages, k))
        remaining -= k
    parts = await asyncio.gather(*chunks)
    return [c for p in parts for c in p]


async def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]

    with open("data/obfuscation_gpt54/input_questions.json") as f:
        all_qs = json.load(f)
    qmap = {(q["batch"], q["idx"]): q for q in all_qs}

    out_dir = Path("exp/direct_answer_gpt54_cot")
    out_dir.mkdir(parents=True, exist_ok=True)

    for batch, idx in TARGETS:
        q = qmap[(batch, idx)]
        if not q["swap"]:
            answer_a, answer_b = q["correct_answer"], q["incorrect_answer"]
            correct_letter = "A"
        else:
            answer_a, answer_b = q["incorrect_answer"], q["correct_answer"]
            correct_letter = "B"
        messages = build_messages(q["question"], answer_a, answer_b, q["story"])
        print(f"[{batch}/{idx}] generating {N_SAMPLES} samples…")
        responses = await call_gpt54(messages, N_SAMPLES)
        parsed = [parse_answer(r) for r in responses]
        out = {
            "batch": batch, "idx": idx,
            "story_title": q["story_title"],
            "question": q["question"],
            "answer_a": answer_a, "answer_b": answer_b,
            "correct_letter": correct_letter,
            "swap": q["swap"],
            "responses": responses,
            "parsed": parsed,
            "n_correct": sum(1 for p in parsed if p == correct_letter),
        }
        p = out_dir / f"{batch}_{idx}.json"
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
        n_cor = out["n_correct"]
        print(f"  saved {p}  {n_cor}/{N_SAMPLES} correct (correct_letter={correct_letter})")
        print(f"  parsed: {parsed}")


if __name__ == "__main__":
    asyncio.run(main())
