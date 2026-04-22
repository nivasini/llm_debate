#!/usr/bin/env python3
"""extend_v3_66_n50.py

Extend the v3/66 case-study cache to n=50 per mode by generating
additional samples and appending to existing caches in
data/obfuscation_gpt54/case_v3_66/. Skips DA no-CoT.

Current state (before running):
  direct_answer_cot_n8.json            : 8  → need 42 more
  v3_66_assigned_correct_gen/judge     : 32 → need 18 more
  v3_66_assigned_incorrect_gen/judge   : 32 → need 18 more
  v3_66_free_choice_gen/judge          : 32 → need 18 more
  v3_66_defendability_gen              : 32 → need 18 more  (no judge)
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.obfuscation_candidates_gpt54 import (
    build_assigned_messages,
    build_free_choice_messages,
    build_defendability_messages,
    build_judge_messages,
    verify_quotes,
    extract_argument,
    extract_choice,
    extract_judge_answer,
    CONSULTANT_MODEL, JUDGE_MODEL,
    CONSULTANT_TEMPERATURE, JUDGE_TEMPERATURE,
    CONSULTANT_MAX_TOKENS, JUDGE_MAX_TOKENS, DEFENDABILITY_MAX_TOKENS,
    MAX_N_PER_CALL,
)
from scripts.direct_answer_gpt54_cot import (
    SYSTEM as DA_COT_SYSTEM,
    build_messages as build_da_cot_messages,
    parse_answer as parse_da_cot_answer,
    MAX_COMPLETION_TOKENS as DA_COT_MAX_TOKENS,
)

TARGET_BATCH = "v3"
TARGET_IDX = 66
TARGET_N = 50
CASE_DIR = Path("data/obfuscation_gpt54/case_v3_66")


async def call_gpt54(messages, n, temperature, max_tokens, model):
    chunks = []
    remaining = n
    while remaining > 0:
        k = min(MAX_N_PER_CALL, remaining)
        chunks.append(openai.ChatCompletion.acreate(
            model=model, messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            n=k,
        ))
        remaining -= k
    parts = await asyncio.gather(*chunks)
    out = []
    for p in parts:
        out.extend(c["message"]["content"] for c in p["choices"])
    return out


async def extend_da_cot(question_data):
    path = CASE_DIR / "direct_answer_cot_n8.json"
    out_path = CASE_DIR / "direct_answer_cot_n50.json"
    with open(path) as f:
        existing = json.load(f)
    n_existing = len(existing["responses"])
    n_new = TARGET_N - n_existing
    print(f"[DA-CoT] existing={n_existing}, generating {n_new} more")

    messages = build_da_cot_messages(
        existing["question"], existing["answer_a"], existing["answer_b"],
        question_data["story"],
    )
    new_responses = await call_gpt54(
        messages, n=n_new,
        temperature=0.8,
        max_tokens=DA_COT_MAX_TOKENS,
        model="gpt-5.4",
    )
    new_parsed = [parse_da_cot_answer(r) for r in new_responses]

    merged = {
        **existing,
        "responses": existing["responses"] + new_responses,
        "parsed": existing["parsed"] + new_parsed,
        "n_correct": sum(1 for p in (existing["parsed"] + new_parsed)
                         if p == existing["correct_letter"]),
    }
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[DA-CoT] saved {out_path}  "
          f"(n={len(merged['responses'])}, correct={merged['n_correct']})")


async def extend_pipeline(cond, question_data):
    gen_path = CASE_DIR / f"v3_66_{cond}_gen.json"
    judge_path = CASE_DIR / f"v3_66_{cond}_judge.json"
    out_gen = CASE_DIR / f"v3_66_{cond}_gen_n50.json"
    out_judge = CASE_DIR / f"v3_66_{cond}_judge_n50.json"

    with open(gen_path) as f:
        gen = json.load(f)
    n_existing = len(gen["candidates"])
    n_new = TARGET_N - n_existing
    print(f"[{cond}] existing={n_existing}, generating {n_new} more")

    correct = question_data["correct_answer"]
    incorrect = question_data["incorrect_answer"]
    story = question_data["story"]
    swap = gen["swap"]
    correct_pos = gen["correct_pos"]
    answer_a, answer_b = (
        (correct, incorrect) if not swap else (incorrect, correct)
    )

    if cond == "assigned_correct":
        messages = build_assigned_messages(
            question_data["question"], correct, incorrect, story)
        max_toks = CONSULTANT_MAX_TOKENS
    elif cond == "assigned_incorrect":
        messages = build_assigned_messages(
            question_data["question"], incorrect, correct, story)
        max_toks = CONSULTANT_MAX_TOKENS
    elif cond == "free_choice":
        messages = build_free_choice_messages(
            question_data["question"], answer_a, answer_b, story)
        max_toks = CONSULTANT_MAX_TOKENS
    elif cond == "defendability":
        messages = build_defendability_messages(
            question_data["question"], answer_a, answer_b, story)
        max_toks = DEFENDABILITY_MAX_TOKENS
    else:
        raise ValueError(cond)

    new_cands = await call_gpt54(
        messages, n=n_new,
        temperature=CONSULTANT_TEMPERATURE,
        max_tokens=max_toks,
        model=CONSULTANT_MODEL,
    )

    new_choices = None
    new_sides = None
    if cond in ("free_choice", "defendability"):
        new_choices = [extract_choice(c) for c in new_cands]
        new_sides = [
            "unknown" if ch is None else
            ("correct" if ch == correct_pos else "incorrect")
            for ch in new_choices
        ]

    merged = {
        **gen,
        "candidates": gen["candidates"] + new_cands,
        "choices": (
            (gen.get("choices") or []) + new_choices
            if new_choices is not None else gen.get("choices")
        ),
        "sides": (
            (gen.get("sides") or []) + new_sides
            if new_sides is not None else gen.get("sides")
        ),
    }
    with open(out_gen, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[{cond}] saved {out_gen}  (n={len(merged['candidates'])})")

    # Judge step (skip for defendability)
    if cond == "defendability":
        return

    with open(judge_path) as f:
        judge = json.load(f)

    async def judge_one(cand):
        arg = extract_argument(cand)
        if not arg:
            return {"argument": "", "verified_argument": "",
                    "judge_response": "", "judge_answer": None,
                    "error": "empty_argument", "is_correct": None}
        verified = verify_quotes(arg, story)
        jmsgs = build_judge_messages(
            question_data["question"], answer_a, answer_b, verified)
        try:
            outs = await call_gpt54(
                jmsgs, n=1,
                temperature=JUDGE_TEMPERATURE,
                max_tokens=JUDGE_MAX_TOKENS,
                model=JUDGE_MODEL,
            )
        except Exception as e:
            return {"argument": arg, "verified_argument": verified,
                    "judge_response": "", "judge_answer": None,
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "is_correct": None}
        text = outs[0]
        ans = extract_judge_answer(text)
        return {
            "argument": arg, "verified_argument": verified,
            "judge_response": text, "judge_answer": ans,
            "is_correct": (None if ans is None else ans == correct_pos),
            "error": None,
        }

    print(f"[{cond}] judging {len(new_cands)} new candidates")
    new_results = await asyncio.gather(*[judge_one(c) for c in new_cands])
    merged_judge = {
        **judge,
        "results": judge["results"] + new_results,
    }
    with open(out_judge, "w") as f:
        json.dump(merged_judge, f, indent=2)
    print(f"[{cond}] saved {out_judge}  "
          f"(n={len(merged_judge['results'])})")


async def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]

    with open("data/obfuscation_gpt54/input_questions.json") as f:
        all_qs = json.load(f)
    q = next(x for x in all_qs
             if x["batch"] == TARGET_BATCH and x["idx"] == TARGET_IDX)

    await extend_da_cot(q)
    for cond in ("assigned_correct", "assigned_incorrect",
                 "free_choice", "defendability"):
        await extend_pipeline(cond, q)


if __name__ == "__main__":
    asyncio.run(main())
