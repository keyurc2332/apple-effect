"""
Text Cleaning Pipeline
======================
Transforms raw YouTube comments into clean, analysis-ready text.

Rules:
  - Remove URLs, HTML tags, excessive punctuation
  - Filter out spam, extremely short/long comments
  - Deduplicate (exact + near-duplicate)
  - Detect language (keep English only by default)
  - Normalise product mentions to canonical names
"""

import re
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# ── Product name normalisation ───────────────────────────────────────────────
PRODUCT_ALIASES = {
    r"\biphone\s?duo\b":          "iPhone Duo",
    r"\bifold\b":                 "iPhone Duo",     # community shorthand
    r"\bappe?l?e?\s?fold(?:able)?\b": "iPhone Duo",
    r"\biphone\s?18\s?pro\s?max\b":   "iPhone 18 Pro Max",
    r"\biphone\s?18\s?pro\b":         "iPhone 18 Pro",
    r"\biphone\s?18\b":               "iPhone 18",
    r"\bgalaxy\s?z?\s?fold\s?6\b":    "Galaxy Z Fold 6",
    r"\bsamsung\s?fold\b":            "Galaxy Z Fold 6",
    r"\ba20\s?pro\b":                 "A20 Pro chip",
}

# Spam signal patterns
SPAM_PATTERNS = [
    r"(?:subscribe|follow|check out my channel)",
    r"(?:first!|first comment)",
    r"(?:https?://|www\.)",                     # pure-link comments
    r"^(?:lol|lmao|haha|😂)+$",                # reaction-only
]

MIN_WORDS   = 4
MAX_WORDS   = 500
LANG_DETECT = True          # requires langdetect if True


def clean_text(text: str) -> str:
    """Apply cleaning rules to a single string."""
    # HTML entities
    text = re.sub(r"&amp;",  "&",  text)
    text = re.sub(r"&lt;",   "<",  text)
    text = re.sub(r"&gt;",   ">",  text)
    text = re.sub(r"&quot;", '"',  text)
    text = re.sub(r"&#39;",  "'",  text)
    # HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Excessive punctuation (3+ repeated chars)
    text = re.sub(r"([!?.,-])\1{2,}", r"\1", text)
    # Emojis → space (keeps surrounding words intact)
    text = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002702-\U000027B0\U000024C2-\U0001F251]+",
        " ", text
    )
    # Newlines / tabs → space
    text = re.sub(r"[\n\r\t]+", " ", text)
    # Collapse whitespace
    text = re.sub(r" {2,}", " ", text).strip()
    return text


def normalise_products(text: str) -> str:
    text_lower = text.lower()
    for pattern, canonical in PRODUCT_ALIASES.items():
        text_lower = re.sub(pattern, canonical, text_lower, flags=re.IGNORECASE)
    return text_lower


def is_spam(text: str) -> bool:
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def is_too_short_or_long(text: str) -> bool:
    n = len(text.split())
    return n < MIN_WORDS or n > MAX_WORDS


def detect_english(text: str) -> bool:
    try:
        from langdetect import detect, LangDetectException
        try:
            return detect(text) == "en"
        except LangDetectException:
            return True   # keep if detection fails
    except ImportError:
        return True       # keep all if langdetect not installed


def clean_comments_df(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Full cleaning pipeline on a comments DataFrame.
    Returns cleaned DataFrame with new column 'text_clean'.
    """
    original_count = len(df)
    df = df.copy()

    # 1. Drop missing text
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip() != ""]

    # 2. Clean text
    df["text_clean"] = df[text_col].astype(str).apply(clean_text)

    # 3. Length filter
    df = df[~df["text_clean"].apply(is_too_short_or_long)]

    # 4. Spam filter
    df = df[~df["text_clean"].apply(is_spam)]

    # 5. Exact deduplication
    df = df.drop_duplicates(subset=["text_clean"])

    # 6. Language filter (optional — can be slow on large datasets)
    if LANG_DETECT:
        log.info("Running language detection (may be slow)…")
        df = df[df["text_clean"].apply(detect_english)]

    # 7. Near-duplicate detection (simple: collapse identical first 80 chars)
    df["_prefix"] = df["text_clean"].str[:80]
    df = df.drop_duplicates(subset=["_prefix"])
    df = df.drop(columns=["_prefix"])

    # 8. Normalise product names
    df["text_clean"] = df["text_clean"].apply(normalise_products)

    removed = original_count - len(df)
    log.info(f"Cleaning: {original_count} → {len(df)} comments  ({removed} removed, {removed/max(original_count,1)*100:.1f}%)")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # quick test
    test_comments = [
        "The camera on the iPhone Duo looks incredible compared to my old phone!",
        "Way too expensive at $2000. No way.",
        "check out my channel for more apple content!!!!",
        "lol",
        "I might get one if they drop the price to $1500 tbh",
        "https://youtu.be/some_link",
        "First!!",
        "The A20 Pro chip makes the ifold blazing fast.",
    ]
    df_test = pd.DataFrame({"text": test_comments})
    result = clean_comments_df(df_test)
    print(result[["text", "text_clean"]].to_string())
