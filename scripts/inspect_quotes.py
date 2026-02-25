#!/usr/bin/env python3
"""
inspect_quotes.py — Analyze verified vs unverified quote rates for correct
vs incorrect consultants, and how they relate to judge accuracy.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rollouts.utils import TranscriptConfig
from web.backend.services.parser import TranscriptParser

SOURCE_DIR = Path("./exp/bon_sweep/bon16")
RESULTS_DIR = Path("./exp/multiseed_comparison")


def extract_argument(text):
    if "<argument>" not in text:
        return text
    return text.split("<argument>")[1].split("</argument>")[0]


def verify_argument(argument, story):
    """Verify quotes in an argument against the story. Returns (verified_arg, stats)."""
    temp_transcript = TranscriptConfig(
        index=0,
        swap=False,
        story=story,
        question="",
        answers={"correct": "", "incorrect": ""},
        names={"correct": "C", "incorrect": "C"},
        rounds=[{"incorrect": argument}],
    )
    verified, stats = TranscriptParser.verify_strict(temp_transcript)
    return verified.rounds[0].incorrect, stats


def main():
    correct_csv = pd.read_csv(SOURCE_DIR / "consultancy_correct" / "data0.csv", encoding="utf-8")
    incorrect_csv = pd.read_csv(SOURCE_DIR / "consultancy_incorrect" / "data0.csv", encoding="utf-8")

    # Load logprobs for judge accuracy correlation
    logprobs = json.loads((RESULTS_DIR / "v2_seed42_logprobs.json").read_text())

    # --- Part 1: Quote stats across all candidates ---
    quote_results = []

    for side, csv_df in [("correct", correct_csv), ("incorrect", incorrect_csv)]:
        cache_dir = SOURCE_DIR / f"consultancy_{side}" / "cache_data0"
        for idx in range(len(csv_df)):
            row = csv_df.iloc[idx]
            transcript = TranscriptConfig(**json.loads(row["transcript"]))
            story = transcript.story

            cache_file = cache_dir / f"{idx}.json"
            if not cache_file.exists():
                continue
            with open(cache_file) as f:
                cache_data = json.load(f)
            candidates = cache_data[0].get(f"responses_{side}", [])

            for cand_idx, raw in enumerate(candidates):
                arg = extract_argument(raw)
                _, stats = verify_argument(arg, story)
                s = stats["incorrect"]
                quote_results.append({
                    "side": side,
                    "idx": idx,
                    "cand": cand_idx,
                    "n_verified": len(s["verified_quotes"]),
                    "n_unverified": len(s["unverified_quotes"]),
                    "n_total": len(s["verified_quotes"]) + len(s["unverified_quotes"]),
                })

    qdf = pd.DataFrame(quote_results)

    # Summary table
    print("=" * 70)
    print("QUOTE VERIFICATION: CORRECT vs INCORRECT CONSULTANTS")
    print("(across all 16 candidates per question, 50 questions)")
    print("=" * 70)
    rows = []
    for side in ["correct", "incorrect"]:
        s = qdf[qdf["side"] == side]
        rows.append({
            "Side": side,
            "Total args": len(s),
            "Args with quotes": f"{(s['n_total'] > 0).sum()} ({(s['n_total'] > 0).mean():.1%})",
            "Args with unverified": f"{(s['n_unverified'] > 0).sum()} ({(s['n_unverified'] > 0).mean():.1%})",
            "Mean verified/arg": f"{s['n_verified'].mean():.2f}",
            "Mean unverified/arg": f"{s['n_unverified'].mean():.2f}",
            "Unverified rate": f"{s['n_unverified'].sum()}/{s['n_total'].sum()} ({s['n_unverified'].sum()/max(s['n_total'].sum(),1):.1%})",
        })
    print(pd.DataFrame(rows).set_index("Side").to_string())

    # --- Part 2: Effect of unverified quotes on judge accuracy ---
    print()
    print("=" * 70)
    print("EFFECT OF UNVERIFIED QUOTES ON JUDGE ACCURACY (BoN=1, seed 42)")
    print("=" * 70)

    v2_correct_lp = {q["idx"]: q["p_correct"][0] for q in logprobs["correct"]}
    v2_incorrect_lp = {q["idx"]: q["p_correct"][0] for q in logprobs["incorrect"]}

    bon1 = qdf[qdf["cand"] == 0].copy()

    rows = []
    for side, lp_map in [("correct", v2_correct_lp), ("incorrect", v2_incorrect_lp)]:
        s = bon1[bon1["side"] == side].copy()
        s["p_correct"] = s["idx"].map(lp_map)

        has_uq = s[s["n_unverified"] > 0]
        no_uq = s[s["n_unverified"] == 0]

        rows.append({
            "Side": side,
            "With u_quote (n)": len(has_uq),
            "p_correct (with u_quote)": f"{has_uq['p_correct'].mean():.2%}" if len(has_uq) > 0 else "n/a",
            "Without u_quote (n)": len(no_uq),
            "p_correct (without u_quote)": f"{no_uq['p_correct'].mean():.2%}" if len(no_uq) > 0 else "n/a",
            "Difference": f"{has_uq['p_correct'].mean() - no_uq['p_correct'].mean():+.2%}" if len(has_uq) > 0 and len(no_uq) > 0 else "n/a",
        })
    print(pd.DataFrame(rows).set_index("Side").to_string())

    # --- Part 3: BoN selection effect on unverified quote rates ---
    print()
    print("=" * 70)
    print("BON SELECTION EFFECT ON UNVERIFIED QUOTE RATES")
    print("=" * 70)

    # Build lookup
    quote_stats = {}
    for _, row in qdf.iterrows():
        quote_stats[(row["side"], row["idx"], row["cand"])] = {
            "n_verified": row["n_verified"],
            "n_unverified": row["n_unverified"],
        }

    lp_lookup = {}
    for side in ["correct", "incorrect"]:
        for q in logprobs[side]:
            lp_lookup[(side, q["idx"])] = q["p_correct"]

    bon_values = [1, 2, 4, 8, 16]
    table_rows = []

    for side in ["correct", "incorrect"]:
        csv_df = correct_csv if side == "correct" else incorrect_csv
        n = len(csv_df)

        for bon in bon_values:
            n_with_uq = 0
            total = 0

            for idx in range(n):
                p_list = lp_lookup.get((side, idx))
                if p_list is None:
                    continue
                k = min(bon, len(p_list))
                best_cand = int(np.argmax(p_list[:k])) if side == "correct" else int(np.argmin(p_list[:k]))

                qs = quote_stats.get((side, idx, best_cand))
                if qs is None:
                    continue
                total += 1
                if qs["n_unverified"] > 0:
                    n_with_uq += 1

            table_rows.append({
                "Side": side,
                "BoN": bon,
                "Selected with u_quote": f"{n_with_uq}/{total} ({n_with_uq/total:.0%})" if total > 0 else "n/a",
            })

    result_df = pd.DataFrame(table_rows)
    print("\nIncorrect consultant:")
    print(result_df[result_df["Side"] == "incorrect"][["BoN", "Selected with u_quote"]].to_string(index=False))
    print("\nCorrect consultant:")
    print(result_df[result_df["Side"] == "correct"][["BoN", "Selected with u_quote"]].to_string(index=False))


if __name__ == "__main__":
    main()
