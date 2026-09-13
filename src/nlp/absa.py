"""
Aspect-Based Sentiment Analysis (ABSA)
=======================================
Decomposes comments into (aspect, sentiment) pairs.

Pipeline:
  comment → aspect extraction → sentiment per aspect → structured output

Approach:
  Phase A  — Rule/keyword-based aspect extraction (fast, interpretable baseline)
  Phase B  — Transformer-based zero-shot classification for sentiment per aspect
             Uses a cross-encoder like "cross-encoder/nli-deberta-v3-small"

This gives us the baseline → ML comparison required by the project spec.
"""

import re
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ── Aspect taxonomy ──────────────────────────────────────────────────────────
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "foldable_design": [
        "fold", "foldable", "hinge", "crease", "unfold", "book",
        "flip", "open", "inner screen", "outer screen", "dual screen",
    ],
    "display": [
        "screen", "display", "panel", "resolution", "refresh rate",
        "brightness", "amoled", "oled", "lcd", "bezel", "notch",
        "7.6 inch", "5.4 inch",
    ],
    "camera": [
        "camera", "photo", "video", "lens", "zoom", "portrait",
        "night mode", "selfie", "4k", "cinematic", "megapixel",
        "shot", "photography", "film",
    ],
    "battery": [
        "battery", "charging", "charge", "mah", "life", "last all day",
        "drain", "endurance", "usb-c", "wireless charging",
    ],
    "performance": [
        "a20", "chip", "processor", "speed", "fast", "lag", "smooth",
        "performance", "ram", "benchmark", "gaming",
    ],
    "durability": [
        "durable", "durability", "fragile", "break", "scratch",
        "drop", "resistant", "rugged", "tough", "quality build",
    ],
    "ai_siri": [
        "siri", "ai", "apple intelligence", "chatgpt", "gpt",
        "llm", "assistant", "prompt", "generate", "smart",
    ],
    "multitasking": [
        "multitask", "split screen", "stage manager", "productivity",
        "dual app", "side by side", "workflow",
    ],
    "price": [
        "price", "cost", "expensive", "cheap", "value", "worth",
        "dollar", "$", "1999", "2000", "1500", "afford",
    ],
    "ecosystem": [
        "ecosystem", "apple pencil", "mac", "ipad", "airdrop",
        "icloud", "apple watch", "airpods", "apple pay",
    ],
    "comparison_samsung": [
        "samsung", "galaxy", "android", "z fold", "pixel",
        "competitor", "versus", "vs",
    ],
    "upgrade": [
        "upgrade", "switch", "trade in", "worth upgrading",
        "keep my", "old phone", "replace", "coming from",
    ],
}

ASPECT_LABELS = list(ASPECT_KEYWORDS.keys())


# ── Rule-based aspect extraction ─────────────────────────────────────────────
def extract_aspects_rule_based(text: str) -> list[str]:
    """Return list of aspects mentioned in the text."""
    text_lower = text.lower()
    found = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(aspect)
    return found


# ── Zero-shot sentiment per aspect ───────────────────────────────────────────
_classifier = None

def _get_zero_shot_classifier():
    global _classifier
    if _classifier is None:
        try:
            from transformers import pipeline
            _classifier = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-deberta-v3-small",
                device=-1,  # CPU; set to 0 for GPU
            )
            log.info("Zero-shot classifier loaded")
        except Exception as e:
            log.warning(f"Could not load transformer classifier: {e}")
            _classifier = None
    return _classifier


SENTIMENT_LABELS = ["positive", "negative", "neutral"]


def classify_sentiment_zero_shot(text: str, aspect: str) -> str:
    """Classify sentiment for a specific aspect using zero-shot NLI."""
    clf = _get_zero_shot_classifier()
    if clf is None:
        return classify_sentiment_lexicon(text)   # fallback

    hypothesis_template = f"This text expresses {{}} sentiment about {aspect.replace('_', ' ')}."
    try:
        result = clf(
            text[:512],   # truncate for speed
            candidate_labels=SENTIMENT_LABELS,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )
        return result["labels"][0]
    except Exception as e:
        log.warning(f"Zero-shot classification failed: {e}")
        return "neutral"


# ── Lexicon fallback sentiment ────────────────────────────────────────────────
POSITIVE_WORDS = {
    "amazing", "incredible", "awesome", "great", "love", "perfect",
    "fantastic", "excellent", "impressive", "beautiful", "stunning",
    "best", "worth", "buy", "preorder", "excited",
}
NEGATIVE_WORDS = {
    "terrible", "awful", "bad", "hate", "ugly", "expensive", "overpriced",
    "disappointed", "waste", "skip", "poor", "cheap", "fragile", "break",
    "no way", "never", "worst",
}

def classify_sentiment_lexicon(text: str) -> str:
    text_lower = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


# ── Purchase intent detection ─────────────────────────────────────────────────
INTENT_PATTERNS = {
    "buy": [
        r"\b(?:preorder(?:ed)?|pre-order(?:ed)?)\b",
        r"\b(?:ordered|bought|purchased|getting mine)\b",
        r"\bday one\b",
        r"\bjust (?:ordered|bought)\b",
    ],
    "considering": [
        r"\b(?:thinking about|considering|might get|may get)\b",
        r"\b(?:tempted|on the fence|debating)\b",
        r"\b(?:i(?:'m| am) interested)\b",
    ],
    "not_buying": [
        r"\b(?:no way|won't buy|not buying|not getting|pass|skip)\b",
        r"\b(?:can'?t afford|too expensive for me)\b",
        r"\b(?:sticking with|keeping my)\b",
    ],
    "conditional": [
        r"\b(?:if (?:the )?price|if it (?:was|were)|when the price)\b",
        r"\bwould (?:buy|get|consider) if\b",
        r"\bmaybe (?:next year|v2|wait)\b",
    ],
}

def detect_purchase_intent(text: str) -> str:
    text_lower = text.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return intent
    return "unknown"


# ── Main ABSA function ────────────────────────────────────────────────────────
def analyse_comment(
    text: str,
    use_transformer: bool = False,
) -> dict:
    """
    Full ABSA on a single comment.

    Returns:
        {
          "text": ...,
          "aspects": [{"aspect": "camera", "sentiment": "positive"}, ...],
          "purchase_intent": "considering",
          "n_aspects": 2,
        }
    """
    aspects_found = extract_aspects_rule_based(text)

    aspect_results = []
    for aspect in aspects_found:
        if use_transformer:
            sentiment = classify_sentiment_zero_shot(text, aspect)
        else:
            sentiment = classify_sentiment_lexicon(text)
        aspect_results.append({"aspect": aspect, "sentiment": sentiment})

    return {
        "text":           text,
        "aspects":        aspect_results,
        "purchase_intent": detect_purchase_intent(text),
        "n_aspects":      len(aspect_results),
    }


def analyse_batch(
    texts: list[str],
    use_transformer: bool = False,
    batch_size: int = 32,
) -> list[dict]:
    """Run ABSA on a list of texts with progress logging."""
    results = []
    total = len(texts)
    for i, text in enumerate(texts):
        results.append(analyse_comment(text, use_transformer=use_transformer))
        if (i + 1) % batch_size == 0:
            log.info(f"  ABSA progress: {i+1}/{total}")
    return results


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    samples = [
        "The foldable design is amazing, but there is no way I'm paying $2,000.",
        "Camera looks incredible. Preordered immediately.",
        "Way too expensive. I'm sticking with my iPhone 17 Pro.",
        "The battery life seems disappointing for a $2k phone.",
        "I might buy it if they drop the price to under $1500.",
        "The A20 chip is insane — this thing is faster than my MacBook.",
        "Siri still sucks compared to Gemini. Apple AI is a joke.",
    ]

    for s in samples:
        result = analyse_comment(s, use_transformer=False)
        print(f"\n  TEXT: {s[:80]}")
        for a in result["aspects"]:
            print(f"    {a['aspect']:25s} → {a['sentiment']}")
        print(f"    INTENT: {result['purchase_intent']}")
