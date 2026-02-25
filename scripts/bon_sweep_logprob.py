#!/usr/bin/env python3
"""
bon_sweep_logprob.py — BoN sweep using logprobs from a single gpt-4o-mini judge pass.

For each question and each of 16 cached candidates:
  1. Insert the candidate's argument into the transcript
  2. Randomly assign swap (correct answer = A or B) per question
  3. Call gpt-4o-mini with the judge prompt, extract logprobs for A and B
  4. Compute P(correct) = softmax(logprob_A, logprob_B) for the correct-answer token

The logprobs serve double duty:
  - Preference rating: for correct consultant, rating = P(correct); for incorrect, rating = P(incorrect)
  - Judge accuracy: P(judge correct) = P(correct-answer token) for the selected candidate

Results for all BoN values (1,2,4,8,16) are computed in one pass.
"""

import asyncio
import argparse
import json
import logging
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agents.judge_base import JudgeConfig
from core.agents.judge_quality import JudgeQuality
from core.file_handler import Method
from core.llm_api.base_llm import LanguageModelConfig, PromptConfig, convert_to_prob
from core.llm_api.llm import ModelAPI
from core.rollouts.utils import RolloutConfig, TranscriptConfig
from core.utils import setup_environment

LOGGER = logging.getLogger(__name__)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def extract_argument(text):
    """Extract content between <argument>...</argument> tags."""
    if "<argument>" not in text:
        return text
    return text.split("<argument>")[1].split("</argument>")[0]


def load_judge_config():
    """Load the preference judge config (same prompt as final judge)."""
    config_path = Path("core/config/experiment/judge/consultancy/preference.yaml")
    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return JudgeConfig(
        language_model=LanguageModelConfig(
            model="gpt-4o-mini",
            temperature=0.0,
            top_p=1.0,
            max_tokens=2,
            timeout=120,
        ),
        prompts=PromptConfig(
            word_limit=raw["prompts"]["word_limit"],
            messages=raw["prompts"]["messages"],
        ),
        judge_type="quality",
        use_logprobs=True,
        few_shot_num_samples=0,
    )


def create_judge(api_handler):
    """Create a JudgeQuality instance configured for logprob extraction."""
    config = load_judge_config()
    rollout_config = RolloutConfig(
        rollout_type="quality_sim",
        num_steps=1,
        name1="Consultant",
        name2="Consultant",
        consultant_name="Consultant",
        cross_examiner_name=None,
        judge_name="gpt-4o-mini",
    )
    return JudgeQuality(
        method=Method.consultancy,
        config=config,
        rollout_config=rollout_config,
        api_handler=api_handler,
    )


async def score_candidate(judge, transcript, argument, side, api_handler):
    """Score a single candidate argument. Returns (logprob_A, logprob_B)."""
    new_transcript = deepcopy(transcript).dict()
    new_transcript["rounds"][-1][side] = argument
    new_transcript = TranscriptConfig(**new_transcript)

    messages = judge.construct_messages(new_transcript)

    responses = await api_handler(
        model_ids="gpt-4o-mini",
        prompt=messages,
        temperature=0.0,
        top_p=1.0,
        max_tokens=2,
        logprobs=5,
    )
    assert len(responses) == 1

    logprob_A = responses[0].logprobs[0].get("A", -100)
    logprob_B = responses[0].logprobs[0].get("B", -100)
    return logprob_A, logprob_B


async def process_question(
    judge, api_handler, transcript_json, candidates, side, swap
):
    """Score all candidates for one question. Returns list of (logprob_A, logprob_B)."""
    transcript = TranscriptConfig(**json.loads(transcript_json))

    if swap:
        transcript = judge.swap_transcript(transcript)

    jobs = []
    for candidate in candidates:
        argument = extract_argument(candidate)
        jobs.append(
            score_candidate(judge, transcript, argument, side, api_handler)
        )

    results = await asyncio.gather(*jobs)
    return results


async def run(args):
    source_dir = Path(args.source_dir)
    bon_values = args.bon_values
    seed = args.seed

    setup_environment()

    api_handler = ModelAPI(
        openai_fraction_rate_limit=0.8,
    )
    judge = create_judge(api_handler)

    rng = random.Random(seed)

    all_results = {}
    for side in ["correct", "incorrect"]:
        csv_path = source_dir / f"consultancy_{side}" / "data0.csv"
        cache_dir = source_dir / f"consultancy_{side}" / "cache_data0"

        if not csv_path.exists():
            print(f"ERROR: CSV not found: {csv_path}")
            sys.exit(1)

        df = pd.read_csv(csv_path, encoding="utf-8")
        print(f"\nProcessing {side} consultant ({len(df)} questions)...")

        question_results = []

        for idx, row in df.iterrows():
            cache_file = cache_dir / f"{idx}.json"
            if not cache_file.exists():
                print(f"  WARNING: Cache file missing: {cache_file}")
                continue

            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            response_key = f"responses_{side}"
            raw_candidates = cache_data[0].get(response_key)
            if raw_candidates is None or len(raw_candidates) != 16:
                print(f"  WARNING: Expected 16 candidates in {cache_file}, skipping")
                continue

            # Random swap per question
            swap = rng.choice([True, False])

            # Get transcript from the CSV
            transcript_json = row["transcript"]
            if not isinstance(transcript_json, str) or not transcript_json.strip():
                print(f"  WARNING: No transcript for question {idx}, skipping")
                continue

            # Score all 16 candidates
            logprobs = await process_question(
                judge, api_handler, transcript_json, raw_candidates, side, swap
            )

            # Compute P(A) for each candidate
            p_correct_list = []
            for logprob_A, logprob_B in logprobs:
                probs = softmax([logprob_A, logprob_B])
                # P(correct answer) depends on swap
                if not swap:
                    p_correct = probs[0]  # correct = A
                else:
                    p_correct = probs[1]  # correct = B
                p_correct_list.append(p_correct)

            question_results.append({
                "idx": idx,
                "p_correct_list": p_correct_list,
                "swap": swap,
            })

            if (len(question_results)) % 10 == 0:
                print(f"  Processed {len(question_results)} questions...")

        all_results[side] = question_results
        print(f"  Done: {len(question_results)} questions scored")

    # Compute metrics for each BoN value
    print("\n" + "=" * 80)
    print(f"  BoN SWEEP RESULTS (logprob-based, gpt-4o-mini, seed={seed})")
    print("=" * 80)

    header = f"{'':>30}" + "".join(f"{'BoN=' + str(b):>10}" for b in bon_values)
    sep = "-" * (30 + 10 * len(bon_values))

    print()
    print(header)
    print(sep)

    metrics = {}
    for b in bon_values:
        metrics[b] = {}
        for side in ["correct", "incorrect"]:
            results = all_results[side]
            expected_accuracies = []

            for q in results:
                p_correct_list = q["p_correct_list"]

                # Preference rating for BoN selection
                if side == "correct":
                    # Correct consultant wants to maximize P(correct)
                    ratings = p_correct_list[:b]
                    best_idx = int(np.argmax(ratings))
                else:
                    # Incorrect consultant wants to maximize P(incorrect) = 1 - P(correct)
                    ratings = [1 - p for p in p_correct_list[:b]]
                    best_idx = int(np.argmax(ratings))

                # P(judge correct) for the selected candidate
                p_judge_correct = p_correct_list[best_idx]
                expected_accuracies.append(p_judge_correct)

            metrics[b][side] = np.mean(expected_accuracies)

    # Judge accuracy (= P(judge correct))
    row_str = f"{'E[judge correct] (correct)':>30}"
    for b in bon_values:
        row_str += f"{metrics[b]['correct']:>10.1%}"
    print(row_str)

    row_str = f"{'E[judge correct] (incorrect)':>30}"
    for b in bon_values:
        row_str += f"{metrics[b]['incorrect']:>10.1%}"
    print(row_str)

    print(sep)

    # Win rates
    row_str = f"{'Correct win rate':>30}"
    for b in bon_values:
        row_str += f"{metrics[b]['correct']:>10.1%}"
    print(row_str)

    row_str = f"{'Incorrect win rate':>30}"
    for b in bon_values:
        row_str += f"{1 - metrics[b]['incorrect']:>10.1%}"
    print(row_str)

    print(sep)

    # Overall
    row_str = f"{'Overall E[judge correct]':>30}"
    for b in bon_values:
        avg = (metrics[b]["correct"] + metrics[b]["incorrect"]) / 2
        row_str += f"{avg:>10.1%}"
    print(row_str)

    print()

    # Save detailed results
    output_file = Path(args.output) if args.output else None
    if output_file:
        rows = []
        for b in bon_values:
            for side in ["correct", "incorrect"]:
                rows.append({
                    "bon": b,
                    "side": side,
                    "expected_accuracy": metrics[b][side],
                    "expected_win_rate": metrics[b][side]
                    if side == "correct"
                    else 1 - metrics[b][side],
                })
        results_df = pd.DataFrame(rows)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")

    # Also save per-question logprobs for further analysis
    logprob_output = Path(args.logprob_output) if args.logprob_output else None
    if logprob_output:
        logprob_data = {}
        for side in ["correct", "incorrect"]:
            logprob_data[side] = [
                {"idx": q["idx"], "p_correct": q["p_correct_list"], "swap": q["swap"]}
                for q in all_results[side]
            ]
        logprob_output.parent.mkdir(parents=True, exist_ok=True)
        with open(logprob_output, "w") as f:
            json.dump(logprob_data, f)
        print(f"Saved per-question logprobs to {logprob_output}")

    print(f"\nTotal API cost: ${api_handler.running_cost:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="BoN sweep using logprobs from a single judge pass"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="./exp/bon_sweep/bon16",
        help="Directory with BoN=16 cache (default: ./exp/bon_sweep/bon16)",
    )
    parser.add_argument(
        "--bon-values",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="BoN values to compute (default: 1 2 4 8 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for swap assignment (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./exp/bon_sweep_v2/results.csv",
        help="Output CSV for summary metrics",
    )
    parser.add_argument(
        "--logprob-output",
        type=str,
        default="./exp/bon_sweep_v2/logprobs.json",
        help="Output JSON for per-question logprobs",
    )
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
