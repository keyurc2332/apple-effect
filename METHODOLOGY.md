# Methodology

Detailed technical decisions and their rationale.

## Why YouTube comments?

YouTube is the primary platform for tech product reviews. Unlike Twitter/X (rate-limited, short-form) or Reddit (requires approved API access), YouTube provides long-form consumer opinions with engagement signals (likes, replies) via a well-documented free API. Launch-week videos attract highly motivated commenters — early adopters, comparison shoppers, and brand loyalists — exactly the audience whose perception drives initial sales.

## Why Aspect-Based Sentiment Analysis?

Overall sentiment ("this video is positive") loses signal. A comment like *"camera is incredible but $2000 is insane"* scores as mixed overall — but it contains two precise, actionable signals. ABSA recovers them. For a product intelligence system, aspect-level data is the only data that matters.

## Why rule-based aspect extraction?

Zero-shot NER and LLM-based extraction are more flexible but significantly slower and costlier at scale. Given that the aspect taxonomy is fixed (12 categories defined by the product's feature set), rule-based keyword matching achieves high recall with full interpretability. False positives are filtered by the sentiment classifier downstream.

## Why TF-IDF over embeddings for sentiment?

The ML comparison (Section 5) showed TF-IDF + Logistic Regression outperforming sentence embeddings (F1: 0.899 vs 0.735). This is expected in this domain: sentiment in tech review comments is signalled by highly specific vocabulary ("deal breaker", "overpriced", "day one purchase") that TF-IDF captures as exact features. Sentence embeddings are better at semantic equivalence across paraphrases — a strength that doesn't translate to performance gains when the discriminative signal is lexical.

## Why bootstrap confidence intervals instead of t-tests?

Comment sentiment scores are not normally distributed — they're discrete (-1, 0, +1) with heavy mass at 0 (neutral). The t-test assumes approximate normality. Bootstrap CIs make no distributional assumptions and are valid for any sample size. With 1,000 resamples per aspect, the intervals are stable.

## Why HDBSCAN over K-Means for clustering?

K-Means requires specifying k in advance and assumes spherical clusters. Comment topics don't form spherical clusters — they form irregular dense regions in embedding space. HDBSCAN finds clusters of arbitrary shape, handles noise gracefully (16% noise points is expected for comment data), and doesn't force every point into a cluster.

## Limitations and what they mean

**Selection bias:** YouTube commenters are not representative consumers. They skew younger, more tech-interested, and more opinionated. Purchase intent signals (1.2% "buy", 1.1% "not buying") are rare because most commenters are spectators, not buyers.

**Temporal scope:** Only launch week (Sep 9–13, 2026). Sentiment on a new product typically evolves — initial reactions driven by specs and price give way to hands-on impressions and durability reports. A longitudinal study would produce different findings.

**Aspect coverage:** 12 hand-defined aspects miss emergent topics. The unsupervised clustering (Phase 4) partially addresses this — cluster 6 ("sell a kidney") discovered a price-shock humor category that the taxonomy didn't anticipate.

**Language filtering:** After langdetect + ASCII ratio filtering, some non-English comments likely remain (Hinglish, code-switching). These appear in the "mixed" sentiment bucket more than they affect directional findings.
