#!/usr/bin/env python3
"""
direct_answer.py — Measure whether gpt-4o-mini can identify the correct answer
when simply asked to answer each reading comprehension question (given the story).

This establishes a baseline: does the model "know" the correct answer before any
persuasion game?
"""

import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_api.llm import ModelAPI
from core.utils import setup_environment

MODEL = "gpt-4o-mini"
NUM_RESPONSES = 10
TEMPERATURE = 0.8
NUM_QUESTIONS = 50


def build_direct_answer_messages(question, answer_a, answer_b, story):
    """Build a simple prompt asking the model to pick the correct answer."""
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
    """Extract A or B from a response. Returns 'A', 'B', or None."""
    text = response_text.strip()

    # Check for just "A" or "B"
    if text in ("A", "B"):
        return text

    # Check for "A." or "B." at the start
    if text.startswith("A") and (len(text) == 1 or not text[1].isalpha()):
        return "A"
    if text.startswith("B") and (len(text) == 1 or not text[1].isalpha()):
        return "B"

    # Search for standalone A or B in the text
    match = re.search(r'\b([AB])\b', text)
    if match:
        return match.group(1)

    return None


async def process_question(idx, api_handler, cache_dir):
    """Process a single question: send prompt, collect responses, compute accuracy."""
    cache_path = cache_dir / f"{idx}.json"
    if not cache_path.exists():
        print(f"  WARNING: Cache file missing: {cache_path}")
        return None

    with open(cache_path) as f:
        cache_data = json.load(f)

    t = json.loads(cache_data[0]["transcript"])
    question = t["question"]
    correct_answer = t["answers"]["correct"]
    incorrect_answer = t["answers"]["incorrect"]
    story = t["story"]
    swap = t["swap"]

    # Assign A/B positions based on swap field (matches how questions are presented)
    if not swap:
        answer_a = correct_answer
        answer_b = incorrect_answer
        correct_letter = "A"
    else:
        answer_a = incorrect_answer
        answer_b = correct_answer
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
    num_parsed = sum(1 for p in parsed if p is not None)
    accuracy = num_correct / NUM_RESPONSES

    return {
        "idx": idx,
        "question": question,
        "correct_answer": correct_answer,
        "incorrect_answer": incorrect_answer,
        "swap": swap,
        "correct_letter": correct_letter,
        "raw_responses": raw_responses,
        "parsed_answers": parsed,
        "num_correct": num_correct,
        "num_parsed": num_parsed,
        "accuracy": accuracy,
    }


async def run():
    setup_environment()

    cache_dir = Path("exp/bon_sweep/bon16/consultancy_correct/cache_data0")
    output_dir = Path("exp/direct_answer")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "direct_answer_results.json"

    api_handler = ModelAPI(openai_fraction_rate_limit=0.8)

    # Load existing partial results if available
    all_results = []
    completed_idxs = set()
    if output_path.exists():
        with open(output_path) as f:
            saved_data = json.load(f)
        for entry in saved_data["questions"]:
            all_results.append(entry)
            completed_idxs.add(entry["idx"])
        print(f"Loaded {len(all_results)} previously completed questions, resuming...")

    remaining_idxs = [i for i in range(NUM_QUESTIONS) if i not in completed_idxs]

    if not remaining_idxs:
        print("All questions already processed!")
    else:
        print(f"Processing {len(remaining_idxs)} questions ({NUM_RESPONSES} responses each)...")

    batch_size = 10
    for batch_start in range(0, len(remaining_idxs), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_idxs))
        batch_idxs = remaining_idxs[batch_start:batch_end]
        print(f"\nBatch: questions {batch_idxs}...")

        jobs = [process_question(idx, api_handler, cache_dir) for idx in batch_idxs]
        batch_results = await asyncio.gather(*jobs)

        for result in batch_results:
            if result is not None:
                all_results.append(result)
                print(f"  Q{result['idx']}: accuracy={result['accuracy']:.0%} "
                      f"({result['num_correct']}/{NUM_RESPONSES}), "
                      f"correct={result['correct_letter']}, "
                      f"parsed={result['parsed_answers']}")

        # Save partial progress
        overall_accuracy = np.mean([r["accuracy"] for r in all_results])
        save_data = {
            "overall_accuracy": overall_accuracy,
            "num_questions": len(all_results),
            "num_responses_per_question": NUM_RESPONSES,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "questions": all_results,
        }
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"  [Saved partial progress: {len(all_results)}/{NUM_QUESTIONS} questions, "
              f"overall accuracy so far: {overall_accuracy:.1%}]")

    # Final summary
    overall_accuracy = np.mean([r["accuracy"] for r in all_results])
    accuracies = [r["accuracy"] for r in all_results]

    print("\n" + "=" * 60)
    print("  DIRECT ANSWER BASELINE RESULTS")
    print("=" * 60)
    print(f"  Model:              {MODEL}")
    print(f"  Questions:          {len(all_results)}")
    print(f"  Responses/question: {NUM_RESPONSES}")
    print(f"  Temperature:        {TEMPERATURE}")
    print(f"  Overall accuracy:   {overall_accuracy:.1%}")
    print(f"  Mean accuracy:      {np.mean(accuracies):.3f} +/- {np.std(accuracies):.3f}")
    print(f"  Median accuracy:    {np.median(accuracies):.1%}")
    print(f"  Min accuracy:       {np.min(accuracies):.1%}")
    print(f"  Max accuracy:       {np.max(accuracies):.1%}")
    print(f"  Questions at 100%:  {sum(1 for a in accuracies if a == 1.0)}")
    print(f"  Questions at 0%:    {sum(1 for a in accuracies if a == 0.0)}")
    print(f"\n  Total API cost: ${api_handler.running_cost:.4f}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
