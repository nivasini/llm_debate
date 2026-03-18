#!/usr/bin/env python3
"""
honest_sweep.py — Run the honest consultant on the full 350-question dataset
(150 from v2 + 200 from v3).

The honest consultant is told to determine the correct answer and argue for it.
Generates 8 candidates per question (temperature 0.8, gpt-4o-mini).
Judged by no-context CoT judge (gpt-4o-mini, temperature 0).

Cache structure:
  exp/honest_sweep/cache/{batch}_{idx}_honest_gen.json
  exp/honest_sweep/cache/{batch}_{idx}_honest_judge.json

Output:
  exp/honest_sweep/results.csv
  exp/honest_sweep/details.json
"""

import asyncio
import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_api.llm import ModelAPI
from core.utils import setup_environment

from scripts.consultant_sweep import (
    NUM_CANDIDATES,
    BON_VALUES,
    build_honest_messages,
    detect_chosen_side,
    resolve_side,
    generate_candidates,
    score_candidate_cot,
    build_base_transcript,
    load_cache,
    save_cache,
    extract_argument,
    create_judge,
    judge_is_correct,
)

LOGGER = logging.getLogger(__name__)


def gen_cache_path(cache_dir, batch, idx):
    return cache_dir / f"{batch}_{idx}_honest_gen.json"


def judge_cache_path(cache_dir, batch, idx):
    return cache_dir / f"{batch}_{idx}_honest_judge.json"


def load_questions_from_details(details_path):
    """Load questions from a details.json file."""
    with open(details_path) as f:
        details = json.load(f)
    questions = []
    for q in details:
        questions.append({
            "idx": q["idx"],
            "question": q["question"],
            "correct_answer": q["correct_answer"],
            "incorrect_answer": q["incorrect_answer"],
            "story_title": q.get("story_title", ""),
            "source": q.get("source", ""),
            "swap": q["swap"],
            "transcript": None,  # will load from cache
        })
    return questions


def load_transcript_from_cache(cache_dir, idx, ctype="assigned_correct"):
    """Load transcript from existing gen cache (v2/v3 already have these)."""
    path = cache_dir / f"{idx}_{ctype}_gen.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data
    return None


def load_questions_with_transcripts(details_path, source_cache_dir, questions_csv_path):
    """Load questions and reconstruct transcript data from the source experiment."""
    with open(details_path) as f:
        details = json.load(f)

    # Also load questions.csv for story data if available
    q_df = None
    if questions_csv_path and Path(questions_csv_path).exists():
        q_df = pd.read_csv(questions_csv_path)

    questions = []
    for q in details:
        # We need the transcript for the judge. Load it from the gen cache.
        gen_data = load_cache(source_cache_dir / f"{q['idx']}_assigned_correct_gen.json")
        if gen_data is None:
            LOGGER.warning(f"No gen cache for idx={q['idx']}, skipping")
            continue

        questions.append({
            "idx": q["idx"],
            "question": q["question"],
            "correct_answer": q["correct_answer"],
            "incorrect_answer": q["incorrect_answer"],
            "story_title": q.get("story_title", ""),
            "source": q.get("source", ""),
            "swap": gen_data["swap"],
            "correct_pos": gen_data["correct_pos"],
        })
    return questions


async def generate_honest(api_handler, cache_dir, batch, idx, question, correct_answer,
                          incorrect_answer, story, swap):
    """Generate honest consultant candidates for one question."""
    cache_path = gen_cache_path(cache_dir, batch, idx)
    cached = load_cache(cache_path)
    if cached is not None:
        return cached

    if not swap:
        answer_a, answer_b = correct_answer, incorrect_answer
        correct_pos = "A"
    else:
        answer_a, answer_b = incorrect_answer, correct_answer
        correct_pos = "B"

    messages = build_honest_messages(question, answer_a, answer_b, story)
    candidates = await generate_candidates(api_handler, messages)

    # Detect which side each candidate chose
    sides = []
    for c in candidates:
        raw = detect_chosen_side(c, correct_answer, incorrect_answer)
        sides.append(resolve_side(raw, correct_pos))

    result = {
        "idx": idx,
        "batch": batch,
        "ctype": "honest",
        "swap": swap,
        "correct_pos": correct_pos,
        "candidates": candidates,
        "sides": sides,
    }
    save_cache(cache_path, result)
    return result


async def judge_honest(api_handler, judge, cache_dir, batch, idx, gen_data, transcript):
    """Judge all honest candidates for one question."""
    cache_path = judge_cache_path(cache_dir, batch, idx)
    cached = load_cache(cache_path)
    if cached is not None:
        return cached

    swap = gen_data["swap"]
    base_transcript = build_base_transcript(transcript, swap)

    candidates = gen_data["candidates"]
    sides = gen_data["sides"]

    jobs = []
    for i, c in enumerate(candidates):
        arg = extract_argument(c)
        side = sides[i] if sides[i] in ("correct", "incorrect") else "correct"
        jobs.append(score_candidate_cot(judge, base_transcript, arg, side, swap, api_handler))

    results = await asyncio.gather(*jobs)

    judge_data = {
        "idx": idx,
        "batch": batch,
        "ctype": "honest",
        "answers": [a for a, _ in results],
        "explanations": [t for _, t in results],
        "is_correct": [judge_is_correct(a, swap) for a, _ in results],
    }
    save_cache(cache_path, judge_data)
    return judge_data


def load_batch_questions(batch_label, exp_dir):
    """Load questions for a batch, including story text from the source cache."""
    details_path = exp_dir / "details.json"
    cache_dir = exp_dir / "cache"

    with open(details_path) as f:
        details = json.load(f)

    questions = []
    for q in details:
        idx = q["idx"]
        # Get story from the gen cache (assigned_correct has the full candidate data)
        gen_file = cache_dir / f"{idx}_free_choice_gen.json"
        if not gen_file.exists():
            gen_file = cache_dir / f"{idx}_assigned_correct_gen.json"
        gen_data = load_cache(gen_file)
        if gen_data is None:
            LOGGER.warning(f"No gen cache for {batch_label}/{idx}, skipping")
            continue

        # We need the story text — get it from the free_choice gen's candidates
        # Actually, the story is in the transcript. Let's get it from the judge cache
        # which has the transcript info. Or we can reconstruct from the questions.csv.
        questions.append({
            "idx": idx,
            "batch": batch_label,
            "question": q["question"],
            "correct_answer": q["correct_answer"],
            "incorrect_answer": q["incorrect_answer"],
            "story_title": q.get("story_title", ""),
            "source": q.get("source", ""),
            "swap": gen_data["swap"],
        })
    return questions


def load_story_from_questions_csv(csv_path):
    """Load story text from the questions CSV if it has a transcript column."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    stories = {}
    for idx, row in df.iterrows():
        if "transcript" in df.columns:
            t_json = row.get("transcript", "")
            if isinstance(t_json, str) and t_json.strip():
                t = json.loads(t_json)
                stories[idx] = t.get("story", "")
    return stories


async def run(args):
    setup_environment()
    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    v2_dir = Path(args.v2_dir)
    v3_dir = Path(args.v3_dir)

    api_handler = ModelAPI(openai_fraction_rate_limit=0.8)
    judge = create_judge(api_handler)

    # Load questions from both batches
    print("Loading questions...")

    # For stories, we need to load from the original data sources
    # v2 questions came from bon16 CSV + QuALITY
    # v3 questions came from QuALITY
    # Both have questions.csv with story_title but not the full story text.
    # The story text is needed for the honest consultant prompt.
    # We can get it from the QuALITY dataset directly, or from existing cache files.

    # Strategy: load the story from the gen cache files which contain the full
    # candidate responses. The story was passed to the model, so we need to
    # reconstruct it. Actually, the cleanest way is to load from the details.json
    # which should have the transcript, or from the questions.csv if it has transcripts.

    # Let's check v2's data source
    v2_questions_csv = v2_dir / "questions.csv"
    v3_questions_csv = v3_dir / "questions.csv"

    # Load details for question metadata
    with open(v2_dir / "details.json") as f:
        v2_details = json.load(f)
    with open(v3_dir / "details.json") as f:
        v3_details = json.load(f)

    # We need story text. The v2/v3 scripts got stories from QuALITY dataset.
    # Let's reload from QuALITY the same way.
    print("Loading stories from QuALITY dataset...")
    from core.load.quality import (
        parse_dataset,
        dataset_to_questions,
        correct_answer_for_question,
        best_distractor_for_question,
    )

    all_quality_qs = []
    for split in ["train", "dev"]:
        data = await parse_dataset(split)
        all_quality_qs.extend(data)
    quality_questions = dataset_to_questions(all_quality_qs)

    # Build lookup: (question_text, story_title) -> story
    story_lookup = {}
    for q in quality_questions:
        key = (q.question, q.article.title)
        story_lookup[key] = q.article.article

    # Also build lookup by question text alone as fallback
    story_by_question = {}
    for q in quality_questions:
        story_by_question[q.question] = q.article.article

    # Build transcript lookup for the judge (needs full transcript structure)
    # We can reconstruct from details + story
    def make_transcript(q_detail, story):
        return {
            "index": q_detail["idx"],
            "question": q_detail["question"],
            "question_set_id": q_detail.get("question_set_id", ""),
            "story": story,
            "story_title": q_detail.get("story_title", ""),
            "answers": {
                "correct": q_detail["correct_answer"],
                "incorrect": q_detail["incorrect_answer"],
            },
            "names": {
                "correct": "Consultant",
                "incorrect": "Consultant",
                "cross_examiner": None,
                "judge": None,
            },
            "swap": False,
            "rollout_type": "quality_sim",
            "rounds": [{"correct": "", "incorrect": "", "judge": ""}],
            "responses": [],
            "extra": {},
        }

    # Assemble all questions with stories
    all_questions = []
    missing = 0

    for batch_label, details, source_cache in [
        ("v2", v2_details, v2_dir / "cache"),
        ("v3", v3_details, v3_dir / "cache"),
    ]:
        for q in details:
            # Get story
            story = story_lookup.get((q["question"], q.get("story_title", "")))
            if story is None:
                story = story_by_question.get(q["question"])
            if story is None:
                missing += 1
                LOGGER.warning(f"No story for {batch_label}/{q['idx']}: {q['question'][:60]}...")
                continue

            # Get swap from existing gen cache
            gen_data = load_cache(source_cache / f"{q['idx']}_assigned_correct_gen.json")
            if gen_data is None:
                gen_data = load_cache(source_cache / f"{q['idx']}_free_choice_gen.json")
            if gen_data is None:
                LOGGER.warning(f"No gen cache for {batch_label}/{q['idx']}")
                continue

            transcript = make_transcript(q, story)

            all_questions.append({
                "idx": q["idx"],
                "batch": batch_label,
                "question": q["question"],
                "correct_answer": q["correct_answer"],
                "incorrect_answer": q["incorrect_answer"],
                "story_title": q.get("story_title", ""),
                "source": q.get("source", ""),
                "story": story,
                "swap": gen_data["swap"],
                "transcript": transcript,
            })

    if missing:
        print(f"WARNING: {missing} questions missing stories")
    print(f"Loaded {len(all_questions)} questions ({sum(1 for q in all_questions if q['batch'] == 'v2')} v2, "
          f"{sum(1 for q in all_questions if q['batch'] == 'v3')} v3)")

    # Phase 1: Generate honest candidates
    print("\n--- Phase 1: Generate honest candidates ---")
    gen_jobs = []
    gen_job_keys = []
    for q in all_questions:
        cp = gen_cache_path(cache_dir, q["batch"], q["idx"])
        if cp.exists():
            continue
        gen_jobs.append(
            generate_honest(
                api_handler, cache_dir, q["batch"], q["idx"],
                q["question"], q["correct_answer"], q["incorrect_answer"],
                q["story"], q["swap"],
            )
        )
        gen_job_keys.append((q["batch"], q["idx"]))

    total_gen = len(all_questions)
    cached_gen = total_gen - len(gen_jobs)
    if gen_jobs:
        print(f"  {len(gen_jobs)} generation tasks ({cached_gen} cached)...")
        batch_size = args.batch_size
        for i in range(0, len(gen_jobs), batch_size):
            batch = gen_jobs[i:i + batch_size]
            await asyncio.gather(*batch)
            done = min(i + batch_size, len(gen_jobs))
            print(f"  Generated {done}/{len(gen_jobs)}")
    else:
        print("  All generation cached!")

    # Phase 2: Judge honest candidates
    print("\n--- Phase 2: Judge honest candidates ---")
    judge_jobs = []
    for q in all_questions:
        cp = judge_cache_path(cache_dir, q["batch"], q["idx"])
        if cp.exists():
            continue
        gen_data = load_cache(gen_cache_path(cache_dir, q["batch"], q["idx"]))
        if gen_data is None:
            LOGGER.warning(f"No gen cache for {q['batch']}/{q['idx']}")
            continue
        judge_jobs.append(
            judge_honest(
                api_handler, judge, cache_dir, q["batch"], q["idx"],
                gen_data, q["transcript"],
            )
        )

    total_judge = len(all_questions)
    cached_judge = total_judge - len(judge_jobs)
    if judge_jobs:
        print(f"  {len(judge_jobs)} judging tasks ({cached_judge} cached)...")
        batch_size = args.batch_size
        for i in range(0, len(judge_jobs), batch_size):
            batch = judge_jobs[i:i + batch_size]
            await asyncio.gather(*batch)
            done = min(i + batch_size, len(judge_jobs))
            print(f"  Judged {done}/{len(judge_jobs)}")
    else:
        print("  All judging cached!")

    # Phase 3: Assemble results
    print("\n--- Phase 3: Assemble results ---")
    all_results = []
    for q in all_questions:
        gen_data = load_cache(gen_cache_path(cache_dir, q["batch"], q["idx"]))
        judge_data = load_cache(judge_cache_path(cache_dir, q["batch"], q["idx"]))
        if gen_data is None or judge_data is None:
            continue

        all_results.append({
            "idx": q["idx"],
            "batch": q["batch"],
            "question": q["question"],
            "correct_answer": q["correct_answer"],
            "incorrect_answer": q["incorrect_answer"],
            "story_title": q["story_title"],
            "source": q["source"],
            "swap": q["swap"],
            "sides": gen_data["sides"],
            "answers": judge_data["answers"],
            "is_correct": judge_data["is_correct"],
        })

    print(f"  {len(all_results)}/{len(all_questions)} questions complete")

    # Compute BoN metrics
    print("\n" + "=" * 80)
    print("  HONEST CONSULTANT — BoN Results")
    print("=" * 80)

    # Also load existing results for comparison
    v2_results = pd.read_csv(v2_dir / "results.csv")
    v3_results = pd.read_csv(v3_dir / "results.csv")

    summary_rows = []
    for bon in BON_VALUES:
        # Overall
        for subset_label, subset in [
            ("all", all_results),
            ("v2", [r for r in all_results if r["batch"] == "v2"]),
            ("v3", [r for r in all_results if r["batch"] == "v3"]),
        ]:
            judge_correct_count = 0
            consultant_wins_count = 0
            chose_correct_count = 0
            total = 0

            for q in subset:
                is_correct_list = q["is_correct"][:bon]
                sides = q["sides"][:bon]

                # Honest: pick first candidate where consultant wins
                best_idx = 0
                for i in range(min(bon, len(is_correct_list))):
                    side = sides[i]
                    if side == "correct" and q["is_correct"][i]:
                        best_idx = i
                        break
                    elif side == "incorrect" and not q["is_correct"][i]:
                        best_idx = i
                        break

                judge_got_it = q["is_correct"][best_idx]
                chosen_side = q["sides"][best_idx]
                consultant_won = (
                    (chosen_side == "correct" and judge_got_it)
                    or (chosen_side == "incorrect" and not judge_got_it)
                    or (chosen_side not in ("correct", "incorrect") and judge_got_it)
                )
                chose_correct = chosen_side == "correct"

                if judge_got_it:
                    judge_correct_count += 1
                if consultant_won:
                    consultant_wins_count += 1
                if chose_correct:
                    chose_correct_count += 1
                total += 1

            if total > 0:
                summary_rows.append({
                    "bon": bon,
                    "subset": subset_label,
                    "judge_correct": judge_correct_count / total,
                    "consultant_wins": consultant_wins_count / total,
                    "frac_chose_correct": chose_correct_count / total,
                    "n": total,
                })

    results_df = pd.DataFrame(summary_rows)
    results_path = output_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")

    # Print summary table
    print(f"\n{'BoN':>5} | {'Subset':>6} | {'Judge Correct':>14} | {'Consultant Wins':>16} | {'Chose Correct':>14} | {'N':>5}")
    print("-" * 75)
    for _, r in results_df.iterrows():
        print(f"{int(r['bon']):>5} | {r['subset']:>6} | {r['judge_correct']:>14.1%} | "
              f"{r['consultant_wins']:>16.1%} | {r['frac_chose_correct']:>14.1%} | {int(r['n']):>5}")

    # Comparison with free choice
    print("\n\nComparison: Honest vs Free Choice (consultant win rate)")
    print(f"{'BoN':>5} | {'Honest (all)':>14} | {'Free Choice v2':>15} | {'Free Choice v3':>15}")
    print("-" * 60)
    for bon in BON_VALUES:
        honest_row = results_df[(results_df["bon"] == bon) & (results_df["subset"] == "all")]
        fc_v2 = v2_results[(v2_results["bon"] == bon) & (v2_results["consultant_type"] == "free_choice")]
        fc_v3 = v3_results[(v3_results["bon"] == bon) & (v3_results["consultant_type"] == "free_choice")]
        h_wr = honest_row["consultant_wins"].values[0] if len(honest_row) else 0
        fc2_wr = fc_v2["consultant_wins"].values[0] if len(fc_v2) else 0
        fc3_wr = fc_v3["consultant_wins"].values[0] if len(fc_v3) else 0
        print(f"{bon:>5} | {h_wr:>14.1%} | {fc2_wr:>15.1%} | {fc3_wr:>15.1%}")

    # Save details
    details_path = output_dir / "details.json"
    with open(details_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved details to {details_path}")

    print(f"\nTotal API cost: ${api_handler.running_cost:.4f}")

    # Check for parse failures
    failures = sum(1 for q in all_results for a in q["answers"] if a is None)
    total_cands = sum(len(q["answers"]) for q in all_results)
    if failures:
        print(f"WARNING: {failures}/{total_cands} parse failures")


def main():
    parser = argparse.ArgumentParser(
        description="Honest consultant sweep on the full 350-question dataset"
    )
    parser.add_argument(
        "--v2-dir", type=str, default="./exp/consultant_sweep_v2",
        help="Directory with v2 experiment results",
    )
    parser.add_argument(
        "--v3-dir", type=str, default="./exp/consultant_sweep_v3",
        help="Directory with v3 experiment results",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./exp/honest_sweep",
    )
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Questions to process concurrently per batch")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
