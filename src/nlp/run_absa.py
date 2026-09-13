"""
Phase 3 — Aspect-Based Sentiment Analysis
==========================================
Runs the full ABSA pipeline on cleaned comments.

Steps:
  1. Load cleaned comments
  2. Filter to English-only (stricter pass)
  3. Run aspect extraction + sentiment per comment
  4. Run purchase intent detection
  5. Save structured results
  6. Print the Feature × Sentiment summary table

Run from apple-effect/ root:
    python src/nlp/run_absa.py
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.nlp.absa import analyse_batch, ASPECT_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

PROC = Path("data/processed")
SEP  = "─" * 60


def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ── 1. Load ───────────────────────────────────────────────────────────────────
section("1 / 6  LOAD CLEAN COMMENTS")

df = pd.read_csv(PROC / "comments_with_video_meta.csv")
print(f"  Loaded: {len(df):,} comments")


# ── 2. Stricter English filter ────────────────────────────────────────────────
section("2 / 6  ENGLISH FILTER (stricter pass)")

def is_mostly_english(text: str, threshold: float = 0.7) -> bool:
    """
    Reject comments where fewer than `threshold` fraction of characters
    are ASCII. Catches Hindi, Arabic, Chinese, etc. written in native script,
    AND Hinglish that slipped through langdetect (uses Latin chars but
    non-English vocabulary patterns).
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) >= threshold

def has_enough_english_words(text: str, min_words: int = 4) -> bool:
    """Must have at least min_words space-separated tokens."""
    return len(str(text).split()) >= min_words

mask = (
    df["text_clean"].apply(is_mostly_english) &
    df["text_clean"].apply(has_enough_english_words)
)
df_en = df[mask].copy().reset_index(drop=True)

removed = len(df) - len(df_en)
print(f"  Before: {len(df):,}")
print(f"  After:  {len(df_en):,}  ({removed:,} non-English removed, {removed/len(df)*100:.1f}%)")


# ── 3. Run ABSA ───────────────────────────────────────────────────────────────
section("3 / 6  RUNNING ABSA  (rule-based + lexicon)")
print("  This runs on every comment — takes ~30 seconds…\n")

texts   = df_en["text_clean"].tolist()
results = analyse_batch(texts, use_transformer=False, batch_size=500)

# Flatten into per-aspect rows
aspect_rows = []
for i, res in enumerate(results):
    row = df_en.iloc[i]
    row_base = {
        "comment_idx":    i,
        "video_id":       row.get("video_id", ""),
        "channel":        row.get("channel", ""),
        "text":           row.get("text_clean", ""),
        "like_count":     row.get("like_count", row.get("like_count_x", row.get("like_count_y", 0))),
        "purchase_intent": res["purchase_intent"],
        "n_aspects":      res["n_aspects"],
    }
    if res["aspects"]:
        for asp in res["aspects"]:
            aspect_rows.append({**row_base, "aspect": asp["aspect"], "sentiment": asp["sentiment"]})
    else:
        aspect_rows.append({**row_base, "aspect": "none", "sentiment": "neutral"})

df_aspects = pd.DataFrame(aspect_rows)

# Also save comment-level results (one row per comment, not per aspect)
comment_level = []
for i, res in enumerate(results):
    row = df_en.iloc[i]
    comment_level.append({
        "comment_idx":     i,
        "video_id":        row.get("video_id", ""),
        "channel":         row.get("channel", ""),
        "text":            row.get("text_clean", ""),
        "like_count":      row.get("like_count", row.get("like_count_x", row.get("like_count_y", 0))),
        "aspects_found":   "|".join(a["aspect"] for a in res["aspects"]),
        "purchase_intent": res["purchase_intent"],
        "n_aspects":       res["n_aspects"],
    })
df_comments = pd.DataFrame(comment_level)

print(f"  Comments analysed:    {len(df_en):,}")
print(f"  Aspect mentions found: {len(df_aspects[df_aspects['aspect'] != 'none']):,}")
print(f"  Comments with ≥1 aspect: {(df_comments['n_aspects'] > 0).sum():,}  "
      f"({(df_comments['n_aspects'] > 0).mean()*100:.1f}%)")


# ── 4. Save ───────────────────────────────────────────────────────────────────
section("4 / 6  SAVE")

df_aspects.to_csv(PROC / "absa_aspect_rows.csv",   index=False)
df_comments.to_csv(PROC / "absa_comment_level.csv", index=False)
df_en.to_csv(PROC / "comments_english.csv",         index=False)

print(f"  → data/processed/absa_aspect_rows.csv   ({len(df_aspects):,} rows)")
print(f"  → data/processed/absa_comment_level.csv ({len(df_comments):,} rows)")
print(f"  → data/processed/comments_english.csv   ({len(df_en):,} rows)")


# ── 5. Feature × Sentiment matrix ────────────────────────────────────────────
section("5 / 6  FEATURE × SENTIMENT MATRIX")

df_real = df_aspects[df_aspects["aspect"] != "none"]

pivot = (
    df_real
    .groupby(["aspect", "sentiment"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
)
pivot["total"]   = pivot.sum(axis=1)
pivot["pos_pct"] = (pivot["positive"] / pivot["total"] * 100).round(1)
pivot["neg_pct"] = (pivot["negative"] / pivot["total"] * 100).round(1)
pivot = pivot.sort_values("total", ascending=False)

print(f"\n  {'Aspect':<22} {'Total':>6}  {'Positive':>9}  {'Negative':>9}  {'Signal'}")
print(f"  {'─'*22} {'─'*6}  {'─'*9}  {'─'*9}  {'─'*10}")

for aspect, row in pivot.iterrows():
    if row["pos_pct"] > row["neg_pct"] + 10:
        signal = "🟢 Positive"
    elif row["neg_pct"] > row["pos_pct"] + 10:
        signal = "🔴 Negative"
    else:
        signal = "🟡 Mixed"
    label = aspect.replace("_", " ").title()
    print(f"  {label:<22} {int(row['total']):>6,}  "
          f"{row['pos_pct']:>8.1f}%  {row['neg_pct']:>8.1f}%  {signal}")


# ── 6. Purchase intent summary ────────────────────────────────────────────────
section("6 / 6  PURCHASE INTENT")

intent_counts = df_comments["purchase_intent"].value_counts()
total = len(df_comments)

print()
for intent, count in intent_counts.items():
    bar = "█" * int(count / total * 40)
    print(f"  {intent:<15} {count:>5,}  ({count/total*100:4.1f}%)  {bar}")

# Intent by aspect — what drives buy vs no-buy
print(f"\n  Top aspects mentioned in BUY comments:")
buy_comments = df_comments[df_comments["purchase_intent"] == "buy"]["aspects_found"]
buy_aspects = "|".join(buy_comments).split("|")
buy_counts = pd.Series(buy_aspects).value_counts().head(5)
for asp, cnt in buy_counts.items():
    if asp:
        print(f"    {asp:<25} {cnt:>4}")

print(f"\n  Top aspects mentioned in NOT_BUYING comments:")
no_comments = df_comments[df_comments["purchase_intent"] == "not_buying"]["aspects_found"]
no_aspects = "|".join(no_comments).split("|")
no_counts = pd.Series(no_aspects).value_counts().head(5)
for asp, cnt in no_counts.items():
    if asp:
        print(f"    {asp:<25} {cnt:>4}")

print(f"\n{SEP}")
print("PHASE 3 COMPLETE")
print("  Next: python src/nlp/run_embeddings.py  (topic clustering)")
print(SEP)
