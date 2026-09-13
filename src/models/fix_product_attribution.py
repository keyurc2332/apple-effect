"""
Fix Product Attribution
========================
Old approach: guess product from comment text (unreliable)
New approach: inherit product from source video title/channel

Maps each video to a product category, then comments
inherit that product tag from their video.

Run from apple-effect/ root:
    python src/models/fix_product_attribution.py
"""

import re
import pandas as pd
from pathlib import Path

PROC = Path("data/processed")
RAW  = Path("data/raw/youtube")
SEP  = "─" * 56

def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")

# ── Load ───────────────────────────────────────────────────────────────────────
section("1 / 3  LOAD")

videos   = pd.read_csv(RAW / "videos.csv")
comments = pd.read_csv(PROC / "absa_comment_level.csv")

print(f"  Videos:   {len(videos):,}")
print(f"  Comments: {len(comments):,}")

# ── Map videos to products ────────────────────────────────────────────────────
section("2 / 3  MAP VIDEOS → PRODUCTS")

def classify_video(title: str) -> str:
    t = str(title).lower()

    # Comparison videos — both products discussed
    if any(x in t for x in ["vs", "versus", "compare", "comparison"]):
        if ("duo" in t or "fold" in t) and ("18 pro" in t or "pro max" in t):
            return "comparison"

    # iPhone Duo — foldable
    if any(x in t for x in ["duo", "foldable iphone", "iphone fold"]):
        return "iPhone Duo"

    # iPhone 18 Pro Max
    if "18 pro max" in t or "18 pro+" in t:
        return "iPhone 18 Pro Max"

    # iPhone 18 Pro
    if "18 pro" in t:
        return "iPhone 18 Pro"

    # iPhone 18 general
    if "iphone 18" in t:
        return "iPhone 18"

    # Apple event — covers all
    if any(x in t for x in ["apple event", "apple keynote", "apple launch",
                              "apple september", "iphone launch"]):
        return "multi-product"

    return "general"

videos["product"] = videos["title"].apply(classify_video)

dist = videos["product"].value_counts()
print(f"\n  Video → Product mapping:")
for prod, count in dist.items():
    pct = count / len(videos) * 100
    print(f"    {prod:<25} {count:>4} videos  ({pct:.1f}%)")

# ── Merge product into comments ───────────────────────────────────────────────
section("3 / 3  MERGE INTO COMMENTS")

video_product = videos[["video_id", "product", "title"]].copy()
comments_with_product = comments.merge(video_product, on="video_id", how="left")
comments_with_product["product"] = comments_with_product["product"].fillna("general")

# Save
out_path = PROC / "absa_comment_level.csv"
comments_with_product.to_csv(out_path, index=False)

print(f"\n  Comment → Product distribution:")
prod_dist = comments_with_product["product"].value_counts()
for prod, count in prod_dist.items():
    pct = count / len(comments_with_product) * 100
    print(f"    {prod:<25} {count:>6,} comments  ({pct:.1f}%)")

# Also update aspect rows
aspects = pd.read_csv(PROC / "absa_aspect_rows.csv")
aspects_with_product = aspects.merge(
    video_product[["video_id", "product"]], on="video_id", how="left"
)
aspects_with_product["product"] = aspects_with_product["product"].fillna("general")
aspects_with_product.to_csv(PROC / "absa_aspect_rows.csv", index=False)

print(f"\n  Saved → {out_path}")
print(f"  Saved → data/processed/absa_aspect_rows.csv")

# Quick sanity check — iPhone Duo vs 18 Pro sentiment
print(f"\n  Sanity check — aspect sentiment by product:")
duo = aspects_with_product[aspects_with_product["product"] == "iPhone Duo"]
pro = aspects_with_product[aspects_with_product["product"] == "iPhone 18 Pro"]

for prod_name, prod_df in [("iPhone Duo", duo), ("iPhone 18 Pro", pro)]:
    if len(prod_df) == 0: continue
    pos = (prod_df["sentiment"] == "positive").mean() * 100
    neg = (prod_df["sentiment"] == "negative").mean() * 100
    print(f"    {prod_name:<20} pos={pos:.1f}%  neg={neg:.1f}%  n={len(prod_df):,}")

print(f"\n{SEP}")
print("PRODUCT ATTRIBUTION FIXED")
print("  Videos now drive product labels — not comment text guessing")
print(SEP)
