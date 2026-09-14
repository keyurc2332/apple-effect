# Methodology

Detailed technical decisions and their rationale.

## Why YouTube comments?

YouTube is the primary platform for tech product reviews. Unlike Twitter/X (rate-limited, short-form) or Reddit (requires approved API access), YouTube provides long-form consumer opinions with engagement signals (likes, replies) via a well-documented free API. Launch-week videos attract highly motivated commenters — early adopters, comparison shoppers, and brand loyalists — exactly the audience whose perception drives initial sales.

## Why Aspect-Based Sentiment Analysis?

Overall sentiment ("this video is positive") loses signal. A comment like *"camera is incredible but $2000 is insane"* scores as mixed overall — but it contains two precise, actionable signals. ABSA recovers them. For a product intelligence system, aspect-level data is the only data that matters.

## Why rule-based aspect extraction?

Zero-shot NER and LLM-based extraction are more flexible but significantly slower and costlier at scale. Given that the aspect taxonomy is fixed (12 categories defined by the product's feature set), rule-based keyword matching achieves high recall with full interpretability. False positives are filtered by the sentiment classifier downstream.

## Why TF-IDF over embeddings for sentiment?

Two separate ML experiments were run:

**Automated-label benchmark** (model trained on pipeline-generated labels — measures pipeline consistency):
TF-IDF + LR: F1 = 0.959, Sentence Embeddings + LR: F1 = 0.729. TF-IDF outperformed embeddings because sentiment in tech YouTube comments is driven by specific vocabulary ("deal breaker", "overpriced", "day one purchase") that TF-IDF captures exactly.

**Human-ground-truth benchmark** (5-fold CV on 175 manually labelled comments — primary validity measure):
TF-IDF + LR: F1 = 0.384, κ = 0.184. This is the number that matters. The gap between 0.959 and 0.384 reflects the difficulty of the task — YouTube comment sentiment involves heavy sarcasm, negation, and implicit opinion that the lexicon-based pipeline handles more confidently than warranted.

## Why bootstrap confidence intervals instead of t-tests?

Comment sentiment scores are not normally distributed — they're discrete (-1, 0, +1) with heavy mass at 0 (neutral). The t-test assumes approximate normality. Bootstrap CIs make no distributional assumptions and are valid for any sample size. With 1,000 resamples per aspect, the intervals are stable.

## Why HDBSCAN over K-Means for clustering?

K-Means requires specifying k in advance and assumes spherical clusters. Comment topics don't form spherical clusters — they form irregular dense regions in embedding space. HDBSCAN finds clusters of arbitrary shape, handles noise gracefully (16% noise points is expected for comment data), and doesn't force every point into a cluster.

## Limitations and what they mean

**Selection bias:** YouTube commenters are not representative consumers. They skew younger, more tech-interested, and more opinionated. Purchase intent signals (1.2% "buy", 1.1% "not buying") are rare because most commenters are spectators, not buyers.

**Temporal scope:** Only launch week (Sep 9–13, 2026). Sentiment on a new product typically evolves — initial reactions driven by specs and price give way to hands-on impressions and durability reports. A longitudinal study would produce different findings.

**Aspect coverage:** 12 hand-defined aspects miss emergent topics. The unsupervised clustering (Phase 4) partially addresses this — cluster 6 ("sell a kidney") discovered a price-shock humor category that the taxonomy didn't anticipate.

**Language filtering:** After langdetect + ASCII ratio filtering, some non-English comments likely remain (Hinglish, code-switching). These appear in the "mixed" sentiment bucket more than they affect directional findings.
