"""
Phase 4 — Embeddings + Topic Clustering
=========================================
Converts comments to dense vectors, reduces dimensions with UMAP,
clusters with HDBSCAN, then labels each cluster automatically.

Outputs:
  - data/processed/embeddings.npy       (raw vectors)
  - data/processed/comments_clustered.csv
  - outputs/figures/umap_clusters.html  (interactive Plotly chart)

Run from apple-effect/ root:
    python src/nlp/run_embeddings.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

PROC    = Path("data/processed")
FIGURES = Path("outputs/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

SEP = "─" * 60

def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ── 1. Load ───────────────────────────────────────────────────────────────────
section("1 / 5  LOAD ENGLISH COMMENTS")

df = pd.read_csv(PROC / "comments_english.csv")
# keep only comments that had at least one aspect detected — richer signal
df_asp = pd.read_csv(PROC / "absa_comment_level.csv")
df = df.merge(df_asp[["comment_idx", "aspects_found", "purchase_intent", "n_aspects"]],
              left_index=True, right_on="comment_idx", how="left")

# Use all comments for clustering (aspect-less ones may form their own cluster)
texts = df["text_clean"].fillna("").tolist()
print(f"  Comments to embed: {len(texts):,}")


# ── 2. Embed ──────────────────────────────────────────────────────────────────
section("2 / 5  SENTENCE EMBEDDINGS")

EMB_CACHE = PROC / "embeddings.npy"

if EMB_CACHE.exists():
    log.info("Loading cached embeddings…")
    embeddings = np.load(EMB_CACHE)
    print(f"  Loaded from cache: {embeddings.shape}")
else:
    from sentence_transformers import SentenceTransformer
    log.info("Loading sentence-transformers model (first run downloads ~90MB)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")   # fast, good quality
    log.info(f"Embedding {len(texts):,} comments…")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    np.save(EMB_CACHE, embeddings)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Saved → {EMB_CACHE}")


# ── 3. UMAP ───────────────────────────────────────────────────────────────────
section("3 / 5  UMAP DIMENSIONALITY REDUCTION")

UMAP_CACHE = PROC / "umap_2d.npy"

if UMAP_CACHE.exists():
    log.info("Loading cached UMAP coords…")
    umap_2d = np.load(UMAP_CACHE)
else:
    import umap
    log.info("Running UMAP (takes 1–3 min on 4k comments)…")
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    umap_2d = reducer.fit_transform(embeddings)
    np.save(UMAP_CACHE, umap_2d)

print(f"  2D coords shape: {umap_2d.shape}")
df["umap_x"] = umap_2d[:, 0]
df["umap_y"] = umap_2d[:, 1]


# ── 4. HDBSCAN clustering ────────────────────────────────────────────────────
section("4 / 5  HDBSCAN CLUSTERING")

import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=40,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
)
labels = clusterer.fit_predict(umap_2d)
df["cluster"] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_pct   = (labels == -1).mean() * 100
print(f"  Clusters found:  {n_clusters}")
print(f"  Noise points:    {(labels == -1).sum():,}  ({noise_pct:.1f}%)")

# Auto-label each cluster using top TF-IDF terms
from sklearn.feature_extraction.text import TfidfVectorizer

cluster_labels = {}
print(f"\n  {'Cluster':>8}  {'Size':>6}  Top terms")
print(f"  {'─'*8}  {'─'*6}  {'─'*40}")

for cid in sorted(set(labels)):
    cluster_texts = [texts[i] for i, l in enumerate(labels) if l == cid]
    size = len(cluster_texts)

    if cid == -1:
        cluster_labels[cid] = "Noise"
        print(f"  {'Noise':>8}  {size:>6,}  —")
        continue

    # TF-IDF top terms for this cluster
    try:
        tfidf = TfidfVectorizer(
            max_features=200,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )
        tfidf.fit(cluster_texts)
        scores = tfidf.idf_
        terms  = tfidf.get_feature_names_out()
        # lower idf = more characteristic of this cluster vs corpus
        top_idx = np.argsort(scores)[:8]
        top_terms = [terms[i] for i in top_idx]
    except Exception:
        top_terms = ["(too small)"]

    label = f"Cluster {cid}"
    cluster_labels[cid] = label
    print(f"  {cid:>8}  {size:>6,}  {', '.join(top_terms[:6])}")

df["cluster_label"] = df["cluster"].map(cluster_labels)


# ── 5. Interactive UMAP plot ─────────────────────────────────────────────────
section("5 / 5  INTERACTIVE UMAP VISUALISATION")

import plotly.express as px

# Truncate text for hover
df["hover_text"] = df["text_clean"].str[:120] + "…"

# Only show non-noise for cleaner plot
df_plot = df[df["cluster"] != -1].copy()
df_plot["cluster_str"] = df_plot["cluster"].astype(str)

fig = px.scatter(
    df_plot,
    x="umap_x",
    y="umap_y",
    color="cluster_str",
    hover_data={"hover_text": True, "aspects_found": True,
                "purchase_intent": True, "umap_x": False, "umap_y": False},
    title="Apple Effect — Comment Topic Clusters (UMAP + HDBSCAN)",
    labels={"cluster_str": "Cluster", "umap_x": "", "umap_y": ""},
    template="plotly_dark",
    width=1000,
    height=700,
    opacity=0.7,
)
fig.update_traces(marker=dict(size=4))
fig.update_layout(
    title_font_size=16,
    legend_title_text="Cluster",
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
)

out_html = FIGURES / "umap_clusters.html"
fig.write_html(out_html)
print(f"  Interactive chart saved → {out_html}")
print(f"  Open it in your browser to explore the clusters")

# Save clustered data
df.to_csv(PROC / "comments_clustered.csv", index=False)

# Cluster summary table
summary = (
    df[df["cluster"] != -1]
    .groupby("cluster")
    .agg(
        size=("text_clean", "count"),
        pct_buy=("purchase_intent", lambda x: (x == "buy").mean() * 100),
        pct_not_buying=("purchase_intent", lambda x: (x == "not_buying").mean() * 100),
    )
    .round(1)
    .sort_values("size", ascending=False)
)
summary.to_csv(PROC / "cluster_summary.csv")

print(f"\n  Cluster summary:")
print(f"  {'Cluster':>8}  {'Size':>6}  {'Buy%':>6}  {'NoBuy%':>7}")
print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*7}")
for cid, row in summary.iterrows():
    print(f"  {cid:>8}  {int(row['size']):>6,}  {row['pct_buy']:>5.1f}%  {row['pct_not_buying']:>6.1f}%")

print(f"\n{SEP}")
print("PHASE 4 COMPLETE")
print(f"  Open outputs/figures/umap_clusters.html in your browser")
print(f"  Next: python src/models/run_classifier.py  (ML models)")
print(SEP)
