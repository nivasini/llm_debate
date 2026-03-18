#!/usr/bin/env python3
"""
direct_answer_350.py — Direct answer baseline for all 350 questions.

Asks gpt-4o-mini to directly answer each reading comprehension question
(given the story) without any consultant/judge framework. 8 responses per
question to match the consultant sweep setup.

Uses the same questions as the honest sweep (150 v2 + 200 v3).
Loads stories from QuALITY dataset.

Cache: exp/direct_answer_350/cache/{batch}_{idx}.json
Output: exp/direct_answer_350/results.json
"""

import asyncio
import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_api.llm import ModelAPI
from core.utils import setup_environment

LOGGER = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
NUM_RESPONSES = 8
TEMPERATURE = 0.8


def build_direct_answer_messages(question, answer_a, answer_b, story):
    system_msg = "You are answering a reading comprehension question based on a story."
    user_msg = (
        f"Story:\n{story}\n\n"
        f"Question: {question}\n\n"
        f"A: {answer_a}\n"
        f"B: {answer_b}\n\n"
        "Which answer is correct? Respond with just A or B."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def parse_answer(response_text):
    text = response_text.strip()
    if text in ("A", "B"):
        return text
    if text.startswith("A") and (len(text) == 1 or not text[1].isalpha()):
        return "A"
    if text.startswith("B") and (len(text) == 1 or not text[1].isalpha()):
        return "B"
    match = re.search(r'\b([AB])\b', text)
    if match:
        return match.group(1)
    return None


async def process_question(api_handler, cache_dir, batch, idx, question,
                           correct_answer, incorrect_answer, story, swap):
    cache_path = cache_dir / f"{batch}_{idx}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    if not swap:
        answer_a, answer_b = correct_answer, incorrect_answer
        correct_letter = "A"
    else:
        answer_a, answer_b = incorrect_answer, correct_answer
        correct_letter = "B"

    messages = build_direct_answer_messages(question, answer_a, answer_b, story)

    responses = await api_handler(
        model_ids=MODEL,
        prompt=messages,
        temperature=TEMPERATURE,
        top_p=1.0,
        max_tokens=5,
        n=NUM_RESPONSES,
    )

    raw_responses = [r.completion for r in responses]
    parsed = [parse_answer(r) for r in raw_responses]
    num_correct = sum(1 for p in parsed if p == correct_letter)

    result = {
        "idx": idx,
        "batch": batch,
        "question": question,
        "correct_answer": correct_answer,
        "incorrect_answer": incorrect_answer,
        "swap": swap,
        "correct_letter": correct_letter,
        "raw_responses": raw_responses,
        "parsed_answers": parsed,
        "num_correct": num_correct,
        "num_responses": NUM_RESPONSES,
        "accuracy": num_correct / NUM_RESPONSES,
    }

    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result


async def run(args):
    setup_environment()
    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    v2_dir = Path(args.v2_dir)
    v3_dir = Path(args.v3_dir)

    api_handler = ModelAPI(openai_fraction_rate_limit=0.8)

    # Load stories from QuALITY
    print("Loading stories from QuALITY dataset...")
    from core.load.quality import parse_dataset, dataset_to_questions

    all_quality_qs = []
    for split in ["train", "dev"]:
        data = await parse_dataset(split)
        all_quality_qs.extend(data)
    quality_questions = dataset_to_questions(all_quality_qs)

    story_lookup = {}
    for q in quality_questions:
        story_lookup[(q.question, q.article.title)] = q.article.article
    story_by_question = {}
    for q in quality_questions:
        story_by_question[q.question] = q.article.article

    # Load questions from v2/v3 details + honest sweep for swap values
    honest_cache = Path(args.honest_dir) / "cache"

    all_questions = []
    for batch_label, details_path, source_cache in [
        ("v2", v2_dir / "details.json", v2_dir / "cache"),
        ("v3", v3_dir / "details.json", v3_dir / "cache"),
    ]:
        with open(details_path) as f:
            details = json.load(f)
        for q in details:
            story = story_lookup.get((q["question"], q.get("story_title", "")))
            if story is None:
                story = story_by_question.get(q["question"])
            if story is None:
                continue

            # Use same swap as honest sweep
            hf = honest_cache / f"{batch_label}_{q['idx']}_honest_gen.json"
            if hf.exists():
                with open(hf) as f:
                    hdata = json.load(f)
                swap = hdata["swap"]
            else:
                # Fallback to assigned_correct gen cache
                gen_data_path = source_cache / f"{q['idx']}_assigned_correct_gen.json"
                if gen_data_path.exists():
                    with open(gen_data_path) as f:
                        gdata = json.load(f)
                    swap = gdata["swap"]
                else:
                    swap = False

            all_questions.append({
                "idx": q["idx"],
                "batch": batch_label,
                "question": q["question"],
                "correct_answer": q["correct_answer"],
                "incorrect_answer": q["incorrect_answer"],
                "story_title": q.get("story_title", ""),
                "story": story,
                "swap": swap,
            })

    print(f"Loaded {len(all_questions)} questions")

    # Process
    jobs = []
    job_keys = []
    for q in all_questions:
        cp = cache_dir / f"{q['batch']}_{q['idx']}.json"
        if cp.exists():
            continue
        jobs.append(
            process_question(
                api_handler, cache_dir, q["batch"], q["idx"],
                q["question"], q["correct_answer"], q["incorrect_answer"],
                q["story"], q["swap"],
            )
        )
        job_keys.append((q["batch"], q["idx"]))

    cached = len(all_questions) - len(jobs)
    if jobs:
        print(f"{len(jobs)} questions to process ({cached} cached)...")
        batch_size = args.batch_size
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            await asyncio.gather(*batch)
            done = min(i + batch_size, len(jobs))
            print(f"  Processed {done}/{len(jobs)}")
    else:
        print("All cached!")

    # Assemble results
    all_results = []
    for q in all_questions:
        cp = cache_dir / f"{q['batch']}_{q['idx']}.json"
        if not cp.exists():
            continue
        with open(cp) as f:
            all_results.append(json.load(f))

    print(f"\n{len(all_results)}/{len(all_questions)} questions complete")

    # Summary
    accuracies = [r["accuracy"] for r in all_results]
    overall = np.mean(accuracies)

    print("\n" + "=" * 60)
    print("  DIRECT ANSWER BASELINE (350 questions)")
    print("=" * 60)
    print(f"  Model:              {MODEL}")
    print(f"  Questions:          {len(all_results)}")
    print(f"  Responses/question: {NUM_RESPONSES}")
    print(f"  Temperature:        {TEMPERATURE}")
    print(f"  Overall accuracy:   {overall:.1%}")
    print(f"  Std:                {np.std(accuracies):.3f}")
    print(f"  Questions at 100%:  {sum(1 for a in accuracies if a == 1.0)}")
    print(f"  Questions at 0%:    {sum(1 for a in accuracies if a == 0.0)}")

    # Per-batch
    for bl in ["v2", "v3"]:
        batch_acc = [r["accuracy"] for r in all_results if r["batch"] == bl]
        if batch_acc:
            print(f"  {bl}: {np.mean(batch_acc):.1%} (n={len(batch_acc)})")

    # Save
    output = {
        "overall_accuracy": overall,
        "num_questions": len(all_results),
        "num_responses_per_question": NUM_RESPONSES,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "questions": all_results,
    }
    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")
    print(f"Total API cost: ${api_handler.running_cost:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", default="./exp/consultant_sweep_v2")
    parser.add_argument("--v3-dir", default="./exp/consultant_sweep_v3")
    parser.add_argument("--honest-dir", default="./exp/honest_sweep")
    parser.add_argument("--output-dir", default="./exp/direct_answer_350")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
