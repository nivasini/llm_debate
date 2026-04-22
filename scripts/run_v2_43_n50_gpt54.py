#!/usr/bin/env python3
"""run_v2_43_n50_gpt54.py

Run v2/43 A Gift from Earth at n=50 on gpt-5.4 across the same 5
modes as the v3/66 case study:
  - DA with forced CoT
  - assigned_correct
  - assigned_incorrect
  - free_choice
  - defendability

Fresh n=50 run (not merged with the n=8 from the 350-question sweep,
because that cache lives in GitHub Actions and isn't locally
available). Uses the same prompts as the gpt-5.4 deep run / the v3/66
case study so the two cases are directly comparable.

Output: data/obfuscation_gpt54/case_v2_43/
"""

import asyncio
import json
import os
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
    CONSULTANT_TEMPERATURE, JUDGE_TEMPERATURE,
    CONSULTANT_MAX_TOKENS, JUDGE_MAX_TOKENS, DEFENDABILITY_MAX_TOKENS,
    MAX_N_PER_CALL,
)
from scripts.direct_answer_gpt54_cot import (
    build_messages as build_da_cot_messages,
    parse_answer as parse_da_cot_answer,
    MAX_COMPLETION_TOKENS as DA_COT_MAX_TOKENS,
)

MODEL = "gpt-5.4"
TARGET_BATCH = "v2"
TARGET_IDX = 43
TARGET_N = 50

OUT_DIR = Path("data/obfuscation_gpt54/case_v2_43")
OUT_DIR.mkdir(parents=True, exist_ok=True)


async def call_gpt54(messages, n, temperature, max_tokens):
    chunks = []
    remaining = n
    while remaining > 0:
        k = min(MAX_N_PER_CALL, remaining)
        chunks.append(openai.ChatCompletion.acreate(
            model=MODEL,
            messages=messages,
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


def _exists_and_has_50(path):
    if not path.exists():
        return False
    with open(path) as f:
        d = json.load(f)
    if "responses" in d:
        return len(d["responses"]) >= TARGET_N
    if "candidates" in d:
        return len(d["candidates"]) >= TARGET_N
    if "results" in d:
        return len(d["results"]) >= TARGET_N
    return False


async def run_da_cot(q):
    p = OUT_DIR / "direct_answer_cot_n50.json"
    if _exists_and_has_50(p):
        print(f"[DA-CoT] already complete, skipping")
        return
    if not q["swap"]:
        answer_a, answer_b = q["correct_answer"], q["incorrect_answer"]
        correct_letter = "A"
    else:
        answer_a, answer_b = q["incorrect_answer"], q["correct_answer"]
        correct_letter = "B"
    messages = build_da_cot_messages(q["question"], answer_a, answer_b, q["story"])
    print(f"[DA-CoT] generating {TARGET_N} samples…")
    responses = await call_gpt54(
        messages, n=TARGET_N,
        temperature=0.8, max_tokens=DA_COT_MAX_TOKENS,
    )
    parsed = [parse_da_cot_answer(r) for r in responses]
    n_correct = sum(1 for x in parsed if x == correct_letter)
    out = {
        "batch": TARGET_BATCH, "idx": TARGET_IDX, "model": MODEL,
        "story_title": q["story_title"],
        "question": q["question"],
        "answer_a": answer_a, "answer_b": answer_b,
        "correct_letter": correct_letter,
        "swap": q["swap"],
        "responses": responses, "parsed": parsed, "n_correct": n_correct,
    }
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[DA-CoT] saved (correct={n_correct}/{TARGET_N})")


async def run_condition(cond, q):
    gen_p = OUT_DIR / f"v2_43_{cond}_gen_n50.json"
    judge_p = OUT_DIR / f"v2_43_{cond}_judge_n50.json"
    gen_ok = _exists_and_has_50(gen_p)
    judge_ok = (cond == "defendability") or _exists_and_has_50(judge_p)
    if gen_ok and judge_ok:
        print(f"[{cond}] already complete, skipping")
        return

    correct = q["correct_answer"]
    incorrect = q["incorrect_answer"]
    story = q["story"]
    swap = q["swap"]
    if not swap:
        answer_a, answer_b = correct, incorrect
        correct_pos = "A"
    else:
        answer_a, answer_b = incorrect, correct
        correct_pos = "B"

    if cond == "assigned_correct":
        messages = build_assigned_messages(q["question"], correct, incorrect, story)
        max_toks = CONSULTANT_MAX_TOKENS
    elif cond == "assigned_incorrect":
        messages = build_assigned_messages(q["question"], incorrect, correct, story)
        max_toks = CONSULTANT_MAX_TOKENS
    elif cond == "free_choice":
        messages = build_free_choice_messages(q["question"], answer_a, answer_b, story)
        max_toks = CONSULTANT_MAX_TOKENS
    elif cond == "defendability":
        messages = build_defendability_messages(q["question"], answer_a, answer_b, story)
        max_toks = DEFENDABILITY_MAX_TOKENS
    else:
        raise ValueError(cond)

    if not gen_ok:
        print(f"[{cond}] generating {TARGET_N} samples…")
        cands = await call_gpt54(
            messages, n=TARGET_N,
            temperature=CONSULTANT_TEMPERATURE, max_tokens=max_toks,
        )
        choices = None
        sides = None
        if cond in ("free_choice", "defendability"):
            choices = [extract_choice(c) for c in cands]
            sides = [
                "unknown" if ch is None else
                ("correct" if ch == correct_pos else "incorrect")
                for ch in choices
            ]
        gen = {
            "batch": TARGET_BATCH, "idx": TARGET_IDX, "model": MODEL,
            "condition": cond, "swap": swap, "correct_pos": correct_pos,
            "candidates": cands, "choices": choices, "sides": sides,
            "error": None,
        }
        with open(gen_p, "w") as f:
            json.dump(gen, f, indent=2)
        print(f"[{cond}] gen saved (n={len(cands)})")
    else:
        with open(gen_p) as f:
            gen = json.load(f)
        cands = gen["candidates"]
        print(f"[{cond}] gen already complete")

    if cond == "defendability":
        return

    if judge_ok:
        print(f"[{cond}] judge already complete")
        return

    async def judge_one(cand):
        arg = extract_argument(cand)
        if not arg:
            return {"argument": "", "verified_argument": "",
                    "judge_response": "", "judge_answer": None,
                    "is_correct": None, "error": "empty_argument"}
        verified = verify_quotes(arg, story)
        jmsgs = build_judge_messages(q["question"], answer_a, answer_b, verified)
        try:
            outs = await call_gpt54(
                jmsgs, n=1,
                temperature=JUDGE_TEMPERATURE, max_tokens=JUDGE_MAX_TOKENS,
            )
        except Exception as e:
            return {"argument": arg, "verified_argument": verified,
                    "judge_response": "", "judge_answer": None,
                    "is_correct": None,
                    "error": f"{type(e).__name__}: {str(e)[:200]}"}
        text = outs[0]
        ans = extract_judge_answer(text)
        return {
            "argument": arg, "verified_argument": verified,
            "judge_response": text, "judge_answer": ans,
            "is_correct": (None if ans is None else ans == correct_pos),
            "error": None,
        }

    print(f"[{cond}] judging {len(cands)} candidates…")
    results = await asyncio.gather(*[judge_one(c) for c in cands])
    judge = {
        "batch": TARGET_BATCH, "idx": TARGET_IDX, "model": MODEL,
        "condition": cond, "correct_pos": correct_pos, "swap": swap,
        "results": results,
    }
    with open(judge_p, "w") as f:
        json.dump(judge, f, indent=2)
    nc = sum(1 for r in results if r.get("is_correct") is True)
    print(f"[{cond}] judge saved (correct={nc}/{len(results)})")


async def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    with open("data/obfuscation_gpt54/input_questions.json") as f:
        all_qs = json.load(f)
    q = next(x for x in all_qs
             if x["batch"] == TARGET_BATCH and x["idx"] == TARGET_IDX)
    print(f"Target: {TARGET_BATCH}/{TARGET_IDX}  {q['story_title']}")
    print(f"  Q: {q['question']}")
    print(f"  correct:   {q['correct_answer']}")
    print(f"  incorrect: {q['incorrect_answer']}")
    print(f"  swap: {q['swap']}")

    await run_da_cot(q)
    for cond in ("assigned_correct", "assigned_incorrect",
                 "free_choice", "defendability"):
        await run_condition(cond, q)


if __name__ == "__main__":
    asyncio.run(main())
