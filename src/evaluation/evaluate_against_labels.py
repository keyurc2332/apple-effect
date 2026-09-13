"""
Evaluation Against Manual Labels
==================================
Compares automated ABSA output against human-labelled ground truth.
Run AFTER labelling at least 100 comments with label_tool.py.

Run from apple-effect/ root:
    python src/evaluation/evaluate_against_labels.py
"""

import json
import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, cohen_kappa_score,
    confusion_matrix, f1_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

PROC       = Path("data/processed")
LABEL_FILE = Path("data/processed/manual_labels.json")
REPORTS    = Path("outputs/reports")
REPORTS.mkdir(exist_ok=True)

SEP = "─" * 60

def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")

# ── Load ───────────────────────────────────────────────────────────────────────
if not LABEL_FILE.exists():
    print("No labels found. Run: streamlit run src/evaluation/label_tool.py")
    sys.exit(1)

labels  = json.loads(LABEL_FILE.read_text())
comments = pd.read_csv(PROC / "comments_english.csv")
absa_df  = pd.read_csv(PROC / "absa_comment_level.csv")

print(f"Manual labels loaded: {len(labels)}")

if len(labels) < 50:
    print(f"Only {len(labels)} labels — need at least 50 for meaningful evaluation.")
    print("Keep labelling in the tool, then re-run this script.")
    sys.exit(0)

# ── Build comparison dataframe ─────────────────────────────────────────────────
section("BUILDING COMPARISON DATASET")

rows = []
for idx_str, lbl in labels.items():
    idx = int(idx_str)
    comment_text = comments.iloc[idx]["text_clean"]

    # Get automated prediction for this comment
    auto_row = absa_df[absa_df["comment_idx"] == idx]
    if len(auto_row) == 0:
        continue

    auto = auto_row.iloc[0]
    # primary automated aspect (first in pipe-separated list)
    auto_aspects = str(auto["aspects_found"]).split("|")
    auto_aspect  = auto_aspects[0] if auto_aspects and auto_aspects[0] else "none"
    auto_intent  = auto["purchase_intent"]

    # For sentiment: look up in aspect rows
    asp_rows = pd.read_csv(PROC / "absa_aspect_rows.csv")
    match = asp_rows[(asp_rows["comment_idx"] == idx) &
                     (asp_rows["aspect"] == lbl["aspect"])]
    auto_sentiment = match["sentiment"].iloc[0] if len(match) else "neutral"

    rows.append({
        "idx":            idx,
        "text":           comment_text[:100],
        "human_aspect":   lbl["aspect"],
        "human_sentiment":lbl["sentiment"],
        "human_intent":   lbl["intent"],
        "auto_aspect":    auto_aspect,
        "auto_sentiment": auto_sentiment,
        "auto_intent":    auto_intent,
    })

df = pd.DataFrame(rows)
print(f"Matched {len(df)} comment-label pairs")


# ── 1. Sentiment evaluation ────────────────────────────────────────────────────
section("1 / 3  SENTIMENT CLASSIFICATION")

# Only evaluate on comments where human labelled a clear sentiment
df_sent = df[df["human_sentiment"].isin(["positive","negative","neutral"])].copy()

print(f"\n  Evaluating on {len(df_sent)} comments with clear sentiment labels\n")
print(classification_report(
    df_sent["human_sentiment"],
    df_sent["auto_sentiment"],
    labels=["positive","neutral","negative"],
    digits=3
))

f1_sent = f1_score(df_sent["human_sentiment"], df_sent["auto_sentiment"],
                   average="macro", labels=["positive","neutral","negative"])
kappa   = cohen_kappa_score(df_sent["human_sentiment"], df_sent["auto_sentiment"])
print(f"  Macro F1:    {f1_sent:.3f}")
print(f"  Cohen's κ:   {kappa:.3f}  ", end="")
if   kappa > 0.8:  print("(almost perfect agreement)")
elif kappa > 0.6:  print("(substantial agreement)")
elif kappa > 0.4:  print("(moderate agreement)")
else:              print("(fair agreement — consider improving aspect matching)")

# Confusion matrix
print(f"\n  Confusion matrix (rows=human, cols=auto):")
cm     = confusion_matrix(df_sent["human_sentiment"], df_sent["auto_sentiment"],
                          labels=["positive","neutral","negative"])
labels = ["positive","neutral","negative"]
header = "          " + "  ".join(f"{l:>9}" for l in labels)
print(header)
for i, row in enumerate(cm):
    print(f"  {labels[i]:>9}  " + "  ".join(f"{v:>9}" for v in row))


# ── 2. Purchase intent evaluation ─────────────────────────────────────────────
section("2 / 3  PURCHASE INTENT DETECTION")

df_intent = df[df["human_intent"] != "unknown"].copy()
if len(df_intent) > 10:
    print(f"\n  Evaluating on {len(df_intent)} comments with explicit intent\n")
    print(classification_report(
        df_intent["human_intent"],
        df_intent["auto_intent"],
        digits=3
    ))
    f1_intent = f1_score(df_intent["human_intent"], df_intent["auto_intent"],
                         average="macro", zero_division=0)
    print(f"  Macro F1: {f1_intent:.3f}")
else:
    print(f"  Only {len(df_intent)} non-unknown intent labels — label more to evaluate intent.")
    f1_intent = None


# ── 3. Failure analysis ────────────────────────────────────────────────────────
section("3 / 3  FAILURE ANALYSIS — WHERE THE MODEL GETS IT WRONG")

errors = df[df["human_sentiment"] != df["auto_sentiment"]].copy()
print(f"\n  {len(errors)} sentiment mismatches out of {len(df)} ({len(errors)/len(df)*100:.1f}%)\n")

# Sample errors by type
for (human, auto), group in errors.groupby(["human_sentiment","auto_sentiment"]):
    if len(group) == 0: continue
    print(f"  Human='{human}' → Auto='{auto}'  ({len(group)} cases)")
    sample = group.sample(min(2, len(group)), random_state=42)
    for _, row in sample.iterrows():
        print(f"    → \"{row['text'][:90]}\"")
    print()


# ── Save summary ───────────────────────────────────────────────────────────────
summary = {
    "n_labels":        len(labels),
    "n_evaluated":     len(df),
    "sentiment_f1":    round(f1_sent, 3),
    "cohens_kappa":    round(kappa, 3),
    "intent_f1":       round(f1_intent, 3) if f1_intent else None,
    "error_rate":      round(len(errors)/len(df), 3),
}
import json
(REPORTS / "human_eval_summary.json").write_text(json.dumps(summary, indent=2))

print(f"\n{SEP}")
print("EVALUATION COMPLETE")
print(f"  Sentiment F1 (vs human labels): {f1_sent:.3f}")
print(f"  Cohen's κ:                      {kappa:.3f}")
print(f"  Saved → outputs/reports/human_eval_summary.json")
print(SEP)
