<div align="center">

# Apple Effect
### *ML-powered Product Perception Intelligence*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-34c759?style=flat-square)

**🚀 Live demo:** https://huggingface.co/spaces/Keyur2332/apple-effect


**🚀 Live demo: https://huggingface.co/spaces/Keyur2332/apple-effect**

**Apple announced a new generation of products. I wanted to know what consumers actually thought.**

[Overview](#overview) · [Pipeline](#pipeline) · [Findings](#key-findings) · [Setup](#setup) · [Dashboard](#dashboard) · [Methodology](#methodology)

---

![Dashboard Preview](outputs/figures/dashboard_preview.png)

</div>

---

## Overview

Apple Effect is a **Product Perception Intelligence system** built to analyse public consumer reaction to the **iPhone Duo** (Apple's first foldable, launched September 9, 2026) and the **iPhone 18 Pro / Pro Max**.

Instead of measuring overall sentiment, it decomposes every comment into individual product aspects and classifies how consumers feel about each one separately.

**The core question:**
> *What features are driving positive and negative consumer perception — and which are associated with expressed purchase intent?*

A comment like:
> *"The foldable design is amazing, but there is no way I'm paying $2,000."*

Becomes:
```
Foldable Design  →  Positive
Price            →  Negative
Purchase Intent  →  Not buying
```

That's **Aspect-Based Sentiment Analysis (ABSA)** — and it's the heart of this project.

---

## Pipeline

```
YouTube Data API  ──►  Raw Comments (5,104)
                              │
                    Text Cleaning Pipeline
                    (spam · duplicates · language filter)
                              │
                    English Comments (4,470)
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
        Aspect          Sentiment       Purchase
       Extraction      Classification    Intent
     (rule-based)    (lexicon → ML)    (regex patterns)
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                  Feature × Sentiment Matrix
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             Sentence              Bootstrap
             Embeddings          Confidence
           + UMAP + HDBSCAN       Intervals
           (topic clusters)    (statistical tests)
                    │                   │
                    └─────────┬─────────┘
                              ▼
                     Streamlit Dashboard
```

---

## Key Findings

| Feature | Mentions | Signal | Note |
|---------|----------|--------|------|
| AI / Siri | 1,000 | 🟡 Mixed | Most discussed, least resolved |
| Foldable Design | 920 | 🟡 Mixed | #1 topic for buyers |
| Camera | 693 | 🟢 Positive | Consistently praised |
| Price | 578 | 🟡 Mixed | Debated, not rejected |
| Durability | 100 | 🔴 Negative | Only statistically negative CI |
| Ecosystem | 219 | 🟢 Positive | Strong signal for Apple users |

**The gap between Apple's emphasis and consumer emphasis:**
- Apple pushed: A20 chip, Apple Intelligence, Foldable design
- Consumers discussed: AI/Siri (skeptically), Price (debating value), Durability (worried)

**ML results:**

Two separate experiments were run. The human-ground-truth benchmark is the primary measure of model validity.

### ① Automated-label benchmark
*Measures how well models reproduce the pipeline-generated labels. Not a validity measure — included for transparency.*

| Model | Macro F1 | vs Baseline |
|-------|----------|-------------|
| Majority Baseline | 0.396 | — |
| TF-IDF + Logistic Regression | 0.959 | +0.563 |
| Sentence Embeddings + LR | 0.729 | +0.333 |

### ② Human-ground-truth benchmark ← primary result
*5-fold stratified CV on 175 independently human-labelled comments. This is the number that matters.*

| Model | Macro F1 | Cohen's κ | vs Baseline |
|-------|----------|-----------|-------------|
| Majority Baseline | 0.190 | 0.000 | — |
| **TF-IDF + Logistic Regression** | **0.384** | **0.184** | +0.194 |
| TF-IDF + Random Forest | 0.322 | 0.076 | +0.132 |

**F1 = 0.384** reflects the genuine difficulty of the task. YouTube comment sentiment involves heavy sarcasm, negation, and implicit opinion. Neutral is the hardest class. The gap between 0.959 (auto-label) and 0.384 (human-validated) is itself a finding — the automated pipeline is more confident than warranted. Labels are single-annotator; inter-annotator agreement was not measured (documented limitation).

---

## Setup

### Prerequisites
- Python 3.10+
- YouTube Data API v3 key ([get one here](https://console.cloud.google.com))

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/apple-effect.git
cd apple-effect

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Open `.env` and add your YouTube API key:
```
YOUTUBE_API_KEY=your_key_here
```

### Run

**Phase 0 — Validate data sources:**
```bash
python src/data_collection/validate_sources.py
```

**Phase 1 — Collect data:**
```bash
python src/data_collection/youtube_collector.py
```

**Phase 2 — Clean + EDA:**
```bash
python src/preprocessing/run_eda.py
```

**Phase 3 — ABSA:**
```bash
python src/nlp/run_absa.py
```

**Phase 4 — Embeddings + clustering:**
```bash
python src/nlp/run_embeddings.py
```

**Phase 5 — ML models:**
```bash
python src/models/run_classifier.py
```

**Phase 6 — Statistical analysis:**
```bash
python src/models/run_statistical_analysis.py
```

**Dashboard:**
```bash
streamlit run app/app.py
```

---

## Dashboard

Five pages, Apple-style dark UI:

| Page | What it shows |
|------|--------------|
| **Overview** | KPIs, discussion volume, top videos, key findings |
| **Feature Intelligence** | Sentiment matrix, confidence intervals, drill-down by feature |
| **Consumer Voice** | UMAP cluster map, Apple vs consumer emphasis comparison |
| **Purchase Intent** | Intent distribution, what drives buy vs not-buy |
| **ML Models** | Baseline → TF-IDF → embeddings comparison, predictive words |

---

## Methodology

### Data collection
- **Source:** YouTube Data API v3 (public comments only)
- **Queries:** 9 search terms covering iPhone Duo, iPhone 18 Pro, hands-on, reviews, comparisons
- **Period:** September 9–13, 2026 (launch week)
- **Volume:** 34+ videos, 5,104 raw comments, 4,470 after cleaning

### Text cleaning
1. Remove URLs, HTML entities, excessive punctuation
2. Filter comments under 4 words or over 500 words
3. Remove spam patterns (subscribe requests, reaction-only, pure links)
4. Exact and near-duplicate removal (first 80 characters)
5. Language detection — English only (langdetect + ASCII ratio filter)
6. Product name normalisation (ifold → iPhone Duo, etc.)

### Aspect extraction
Rule-based keyword matching across 12 product aspects:
`foldable_design · display · camera · battery · performance · durability · ai_siri · multitasking · price · ecosystem · comparison_samsung · upgrade`

### Sentiment classification
- **Baseline:** Majority class (F1: 0.397)
- **Model A:** TF-IDF (5,000 features, bigrams) + Logistic Regression (F1: **0.959**)
- **Model B:** Sentence Embeddings (all-MiniLM-L6-v2, 384-dim) + Logistic Regression (F1: 0.729)

### Purchase intent detection
Regex pattern matching across 4 intent classes:
- **Buy** — "preordered", "day one", "just bought"
- **Considering** — "thinking about", "might get", "tempted"
- **Not buying** — "no way", "won't buy", "sticking with"
- **Conditional** — "would buy if", "wait for price drop", "maybe gen 2"

### Statistical analysis
- Bootstrap 95% confidence intervals on aspect sentiment means (n=1,000 resamples)
- Chi-square test for sentiment distribution differences between products
- Cramér's V effect sizes

### Clustering
- Sentence embeddings: `all-MiniLM-L6-v2`
- Dimensionality reduction: UMAP (n_neighbors=15, cosine metric)
- Clustering: HDBSCAN (min_cluster_size=40)
- Result: 19 topic clusters discovered without supervision

### Limitations
- YouTube comments skew toward engaged/opinionated users — not representative of all consumers
- Rule-based aspect extraction misses nuanced or implicit references
- Purchase intent signals are rare (3.3% of comments) — statistical power is limited
- Non-English comments (Hindi, German) required additional filtering; some may have slipped through
- Data is launch-week only — sentiment may shift as reviews mature

---

## Project structure

```
apple-effect/
├── data/
│   ├── raw/              # Raw API data (not committed)
│   ├── processed/        # Cleaned datasets (not committed)
│   └── external/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_topic_modeling.ipynb
│   ├── 04_purchase_intent.ipynb
│   └── 05_statistical_analysis.ipynb
├── src/
│   ├── data_collection/
│   │   ├── validate_sources.py
│   │   ├── youtube_collector.py
│   │   └── trends_collector.py
│   ├── preprocessing/
│   │   ├── cleaner.py
│   │   └── run_eda.py
│   ├── nlp/
│   │   ├── absa.py
│   │   ├── run_absa.py
│   │   └── run_embeddings.py
│   ├── models/
│   │   ├── run_classifier.py
│   │   └── run_statistical_analysis.py
│   └── evaluation/
├── app/
│   └── app.py            # Streamlit dashboard
├── models/               # Saved model files (not committed)
├── outputs/
│   ├── figures/          # Charts and visualisations
│   └── reports/          # Statistical outputs
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Tech stack

| Category | Tools |
|----------|-------|
| Data | YouTube Data API v3, pytrends |
| NLP | Hugging Face Transformers, Sentence Transformers, spaCy |
| ML | scikit-learn, XGBoost |
| Clustering | UMAP, HDBSCAN |
| Stats | SciPy, statsmodels |
| Viz | Plotly, Matplotlib |
| App | Streamlit |

---

## Ethics & data use

- Only public YouTube comments are used — no scraping, no authentication bypass
- Individual usernames are never stored or displayed
- The unit of analysis is the **opinion text**, not the person who wrote it
- Complies with YouTube Data API Terms of Service
- Reddit data was intentionally excluded pending authorized API access

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built as a weekend project to answer one question:
**Apple told us what mattered. What did consumers actually think?**

</div>
