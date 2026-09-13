"""
Phase 2 — EDA + Cleaning
=========================
Loads raw YouTube data, runs the cleaning pipeline,
and prints a full exploratory summary.

Run from the apple-effect/ root:
    python src/preprocessing/run_eda.py
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.cleaner import clean_comments_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

RAW  = Path("data/raw/youtube")
PROC = Path("data/processed")
PROC.mkdir(exist_ok=True)

SEP = "─" * 52

def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ── Load ─────────────────────────────────────────────────────────────────────
section("1 / 6  RAW DATA")

videos   = pd.read_csv(RAW / "videos.csv")
comments = pd.read_csv(RAW / "comments.csv")

print(f"  Videos:   {len(videos):,}")
print(f"  Comments: {len(comments):,}")
print(f"  Date range: {comments['published_at'].min()[:10]} → {comments['published_at'].max()[:10]}")

# ── Video stats ───────────────────────────────────────────────────────────────
section("2 / 6  TOP VIDEOS BY VIEWS")

videos_sorted = videos.sort_values("view_count", ascending=False)
for _, row in videos_sorted.head(10).iterrows():
    print(f"  {int(row['view_count']):>10,}  views  |  {row['title'][:55]}")

section("TOP CHANNELS")
chan = videos["channel"].value_counts().head(10)
for ch, n in chan.items():
    print(f"  {n:>3} videos  |  {ch}")

# ── Comment length ────────────────────────────────────────────────────────────
section("3 / 6  COMMENT LENGTH DISTRIBUTION (raw)")

comments["word_count"] = comments["text"].astype(str).str.split().str.len()
wc = comments["word_count"]
print(f"  Median:  {wc.median():.0f} words")
print(f"  Mean:    {wc.mean():.1f} words")
print(f"  <4 words (would be filtered): {(wc < 4).sum():,}  ({(wc < 4).mean()*100:.1f}%)")
print(f"  >500 words (would be filtered): {(wc > 500).sum():,}  ({(wc > 500).mean()*100:.1f}%)")

# ── Cleaning ──────────────────────────────────────────────────────────────────
section("4 / 6  CLEANING PIPELINE")

comments_clean = clean_comments_df(comments, text_col="text")

removed = len(comments) - len(comments_clean)
print(f"\n  Raw:     {len(comments):,}")
print(f"  Clean:   {len(comments_clean):,}")
print(f"  Removed: {removed:,}  ({removed/len(comments)*100:.1f}%)")

# ── Product mentions ──────────────────────────────────────────────────────────
section("5 / 6  PRODUCT MENTION FREQUENCY (clean)")

txt = comments_clean["text_clean"]

products = {
    "iPhone Duo":        txt.str.contains(r"iphone duo",          case=False, regex=True).sum(),
    "iPhone 18 Pro Max": txt.str.contains(r"18 pro max",          case=False, regex=True).sum(),
    "iPhone 18 Pro":     txt.str.contains(r"18 pro(?! max)",      case=False, regex=True).sum(),
    "Samsung / Android": txt.str.contains(r"samsung|android|galaxy fold", case=False, regex=True).sum(),
    "Price ($)":         txt.str.contains(r"\$|price|expensive|cheap|worth", case=False, regex=True).sum(),
    "Camera":            txt.str.contains(r"camera|photo|video|zoom", case=False, regex=True).sum(),
    "Battery":           txt.str.contains(r"battery|charging|charge", case=False, regex=True).sum(),
    "AI / Siri":         txt.str.contains(r"siri|ai|apple intelligence", case=False, regex=True).sum(),
    "Hinge / Fold":      txt.str.contains(r"hinge|crease|fold|unfold", case=False, regex=True).sum(),
}

max_count = max(products.values())
print()
for term, count in sorted(products.items(), key=lambda x: -x[1]):
    bar = "█" * int(count / max_count * 30)
    pct = count / len(comments_clean) * 100
    print(f"  {term:<22} {count:>5,}  ({pct:4.1f}%)  {bar}")

# ── Sample clean comments ─────────────────────────────────────────────────────
section("6 / 6  SAMPLE CLEAN COMMENTS")

samples = comments_clean.sample(min(10, len(comments_clean)), random_state=42)
for _, row in samples.iterrows():
    print(f"\n  » {row['text_clean'][:120]}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = PROC / "comments_clean.csv"
comments_clean.to_csv(out_path, index=False)

# also save a merged file (comments + video metadata)
merged = comments_clean.merge(
    videos[["video_id", "title", "channel", "view_count", "like_count"]],
    on="video_id",
    how="left"
)
merged.to_csv(PROC / "comments_with_video_meta.csv", index=False)

print(f"\n{SEP}")
print("PHASE 2 COMPLETE")
print(f"  Saved → {out_path}  ({len(comments_clean):,} rows)")
print(f"  Saved → data/processed/comments_with_video_meta.csv")
print(f"\n  Next: run Phase 3 — ABSA")
print(f"         python src/nlp/run_absa.py")
print(SEP)
