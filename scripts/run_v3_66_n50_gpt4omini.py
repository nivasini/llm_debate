#!/usr/bin/env python3
"""run_v3_66_n50_gpt4omini.py

Run v3/66 Bullet with His Name at n=50 on gpt-4o-mini across the same
5 modes as the gpt-5.4 case study (DA-with-forced-CoT, assigned_correct,
assigned_incorrect, free_choice, defendability). Uses the SAME prompts
as the gpt-5.4 deep run so the two are directly comparable.

Note: does NOT reuse the existing n=8 gpt-4o-mini data, because that
data was generated with the older consultant_sweep prompts (no
<choice> tag, weaker judge-blindness framing). We run a fresh n=50
here for clean cross-model comparison.

Output:
  data/obfuscation_gpt54/case_v3_66/gpt4omini/
    direct_answer_cot_n50.json
    v3_66_assigned_correct_gen_n50.json
    v3_66_assigned_correct_judge_n50.json
    ... (same pattern for other conditions)
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
)
from scripts.direct_answer_gpt54_cot import (
    build_messages as build_da_cot_messages,
    parse_answer as parse_da_cot_answer,
)

MODEL = "gpt-4o-mini"
TARGET_BATCH = "v3"
TARGET_IDX = 66
TARGET_N = 50
CONSULTANT_TEMPERATURE = 0.8
JUDGE_TEMPERATURE = 0.0
CONSULTANT_MAX_TOKENS = 1500
JUDGE_MAX_TOKENS = 1500
DEFENDABILITY_MAX_TOKENS = 1500
DA_COT_MAX_TOKENS = 1500
MAX_N_PER_CALL = 8

OUT_DIR = Path("data/obfuscation_gpt54/case_v3_66/gpt4omini")
OUT_DIR.mkdir(parents=True, exist_ok=True)


async def call_model(messages, n, temperature, max_tokens):
    """gpt-4o-mini uses max_tokens (not max_completion_tokens)."""
    chunks = []
    remaining = n
    while remaining > 0:
        k = min(MAX_N_PER_CALL, remaining)
        chunks.append(openai.ChatCompletion.acreate(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=k,
        ))
        remaining -= k
    parts = await asyncio.gather(*chunks)
    out = []
    for p in parts:
        out.extend(c["message"]["content"] for c in p["choices"])
    return out


def _exists_and_has_50(path):
    if not Path(path).exists():
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
        print(f"[DA-CoT] already at n={TARGET_N}, skipping")
        return
    if not q["swap"]:
        answer_a, answer_b = q["correct_answer"], q["incorrect_answer"]
        correct_letter = "A"
    else:
        answer_a, answer_b = q["incorrect_answer"], q["correct_answer"]
        correct_letter = "B"
    messages = build_da_cot_messages(q["question"], answer_a, answer_b, q["story"])
    print(f"[DA-CoT] generating {TARGET_N} samples…")
    responses = await call_model(
        messages, n=TARGET_N,
        temperature=0.8, max_tokens=DA_COT_MAX_TOKENS,
    )
    parsed = [parse_da_cot_answer(r) for r in responses]
    n_correct = sum(1 for p in parsed if p == correct_letter)
    out = {
        "batch": TARGET_BATCH, "idx": TARGET_IDX,
        "model": MODEL,
        "story_title": q["story_title"],
        "question": q["question"],
        "answer_a": answer_a, "answer_b": answer_b,
        "correct_letter": correct_letter,
        "swap": q["swap"],
        "responses": responses,
        "parsed": parsed,
        "n_correct": n_correct,
    }
    p = OUT_DIR / "direct_answer_cot_n50.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[DA-CoT] saved {p}  (correct={n_correct}/{TARGET_N})")


async def run_condition(cond, q):
    gen_path = OUT_DIR / f"v3_66_{cond}_gen_n50.json"
    judge_path = OUT_DIR / f"v3_66_{cond}_judge_n50.json"
    gen_ok = _exists_and_has_50(gen_path)
    judge_ok = (cond == "defendability") or _exists_and_has_50(judge_path)
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

    print(f"[{cond}] generating {TARGET_N} samples…")
    cands = await call_model(
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
        "batch": TARGET_BATCH, "idx": TARGET_IDX,
        "model": MODEL, "condition": cond,
        "swap": swap, "correct_pos": correct_pos,
        "candidates": cands, "choices": choices, "sides": sides,
        "error": None,
    }
    with open(OUT_DIR / f"v3_66_{cond}_gen_n50.json", "w") as f:
        json.dump(gen, f, indent=2)
    print(f"[{cond}] gen saved (n={len(cands)})")

    if cond == "defendability":
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
            outs = await call_model(
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

    print(f"[{cond}] judging {TARGET_N} candidates…")
    results = await asyncio.gather(*[judge_one(c) for c in cands])
    judge = {
        "batch": TARGET_BATCH, "idx": TARGET_IDX,
        "model": MODEL, "condition": cond,
        "correct_pos": correct_pos, "swap": swap,
        "results": results,
    }
    with open(OUT_DIR / f"v3_66_{cond}_judge_n50.json", "w") as f:
        json.dump(judge, f, indent=2)
    n_correct = sum(1 for r in results if r.get("is_correct") is True)
    print(f"[{cond}] judge saved  (judge_correct={n_correct}/{len(results)})")


async def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    with open("data/obfuscation_gpt54/input_questions.json") as f:
        all_qs = json.load(f)
    q = next(x for x in all_qs
             if x["batch"] == TARGET_BATCH and x["idx"] == TARGET_IDX)

    await run_da_cot(q)
    for cond in ("assigned_correct", "assigned_incorrect",
                 "free_choice", "defendability"):
        await run_condition(cond, q)


if __name__ == "__main__":
    asyncio.run(main())
