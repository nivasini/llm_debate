#!/usr/bin/env python3
"""
bon_sweep_labeled_reasoning.py — Inspect the judge's full reasoning text.

Same setup as bon_sweep_labeled.py (labeled context + reflection prompt), but:
  - max_tokens=500 (no logprobs) so we get the judge's full chain-of-thought
  - Only 5 questions (3 incorrect-side, 2 correct-side) for quick inspection
  - 1 candidate per question (the first one)
  - Modified prompt: asks for reasoning before the final A/B answer
"""

import asyncio
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_api.llm import ModelAPI
from core.rollouts.utils import TranscriptConfig
from core.utils import setup_environment

# Reuse helpers from the labeled sweep script
from scripts.bon_sweep_labeled import (
    QuestionData,
    build_labeled_context_block,
    create_judge,
    extract_argument,
)

SOURCE_DIR = Path("./exp/bon_sweep/bon16")
SEED = 42
NUM_CONTEXT = 9
CONTEXT_SEED = 123

# How many questions per side to inspect
NUM_CORRECT = 2
NUM_INCORRECT = 3


def load_all_questions(source_dir, rng):
    """Load question data for both sides (identical to bon_sweep_labeled.py)."""
    import pandas as pd

    all_questions = []
    side_questions = {}

    for side in ["correct", "incorrect"]:
        csv_path = source_dir / f"consultancy_{side}" / "data0.csv"
        cache_dir = source_dir / f"consultancy_{side}" / "cache_data0"

        if not csv_path.exists():
            print(f"ERROR: CSV not found: {csv_path}")
            sys.exit(1)

        df = pd.read_csv(csv_path, encoding="utf-8")
        questions = []

        for idx, row in df.iterrows():
            cache_file = cache_dir / f"{idx}.json"
            if not cache_file.exists():
                continue

            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            response_key = f"responses_{side}"
            raw_candidates = cache_data[0].get(response_key)
            if raw_candidates is None or len(raw_candidates) != 16:
                continue

            swap = rng.choice([True, False])

            transcript_json = row["transcript"]
            if not isinstance(transcript_json, str) or not transcript_json.strip():
                continue

            qd = QuestionData(idx, side, transcript_json, raw_candidates, swap)
            questions.append(qd)
            all_questions.append(qd)

        side_questions[side] = questions
        print(f"Loaded {len(questions)} {side}-side questions")

    return all_questions, side_questions


async def get_reasoning(judge, api_handler, question_data, context_block):
    """Call the judge with max_tokens=500 and a reasoning prompt. Return the full text."""
    transcript = TranscriptConfig(**json.loads(question_data.transcript_json))

    if question_data.swap:
        transcript = judge.swap_transcript(transcript)

    # Use the first candidate only
    candidate = question_data.candidates[0]
    argument = extract_argument(candidate)

    new_transcript = deepcopy(transcript).dict()
    new_transcript["rounds"][-1][question_data.side] = argument
    new_transcript = TranscriptConfig(**new_transcript)

    messages = judge.construct_messages(new_transcript)

    # Prepend context block to the first (user) message
    messages[0]["content"] = context_block + messages[0]["content"]

    # Replace the final "Respond with only A or B." with a reasoning request
    messages[0]["content"] = messages[0]["content"].replace(
        "Respond with only A or B.",
        "First, explain your reasoning step by step. Then give your final answer as A or B on its own line.",
    )

    responses = await api_handler(
        model_ids="gpt-4o-mini",
        prompt=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=500,
    )
    assert len(responses) == 1
    return responses[0].completion, transcript, argument


async def run():
    setup_environment()

    api_handler = ModelAPI(openai_fraction_rate_limit=0.8)
    judge = create_judge(api_handler)

    rng = random.Random(SEED)
    all_questions, side_questions = load_all_questions(SOURCE_DIR, rng)

    # Pick a small subset: 3 incorrect, 2 correct
    selected = []
    for side, count in [("incorrect", NUM_INCORRECT), ("correct", NUM_CORRECT)]:
        pool = side_questions[side]
        selected.extend(pool[:count])

    print(f"\nSelected {len(selected)} questions for reasoning inspection\n")

    # Build context block (use the first selected question's context setup)
    # We use a single shared context block for display, sampled the same way
    # as in the main script.
    side_offset = 0
    context_rng = random.Random(CONTEXT_SEED * 10000 + selected[0].idx + side_offset)
    pool = [
        q
        for q in all_questions
        if not (q.idx == selected[0].idx and q.side == selected[0].side)
    ]
    context_questions = context_rng.sample(pool, min(NUM_CONTEXT, len(pool)))
    context_block = build_labeled_context_block(context_questions, context_rng)

    # Print the context block once
    print("=" * 80)
    print("LABELED CONTEXT BLOCK (shared across all 5 questions)")
    print("=" * 80)
    print(context_block)
    print("=" * 80)
    print()

    # Process each selected question
    for i, qd in enumerate(selected, 1):
        transcript = TranscriptConfig(**json.loads(qd.transcript_json))

        if qd.swap:
            answer_a = transcript.answers.incorrect
            answer_b = transcript.answers.correct
        else:
            answer_a = transcript.answers.correct
            answer_b = transcript.answers.incorrect

        print(f"--- Question {i}/{len(selected)} (idx={qd.idx}, side={qd.side}, swap={qd.swap}) ---")
        print(f"Question: {transcript.question}")
        print(f"A: {answer_a}")
        print(f"B: {answer_b}")
        print(f"Consultant side: {qd.side}")
        print()

        reasoning, _, argument = await get_reasoning(
            judge, api_handler, qd, context_block
        )

        print(f"Consultant argument (first candidate):")
        print(f"  {argument[:200]}{'...' if len(argument) > 200 else ''}")
        print()
        print(f"Judge reasoning:")
        print(reasoning)
        print()
        print("=" * 80)
        print()

    print(f"Total API cost: ${api_handler.running_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(run())
