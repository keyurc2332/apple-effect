"""
Feature Perception Map
======================
Bubble chart: x=discussion volume, y=mean sentiment, size=mentions
The single most powerful visualization in the project.

Run from apple-effect/ root:
    python src/visualization/feature_perception_map.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

PROC    = Path("data/processed")
FIGURES = Path("outputs/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

aspects = pd.read_csv(PROC / "absa_aspect_rows.csv")
ci_df   = pd.read_csv(Path("outputs/reports/aspect_confidence_intervals.csv"))
aspects = aspects[aspects["aspect"] != "none"].copy()
aspects["sentiment_score"] = aspects["sentiment"].map(
    {"positive": 1, "neutral": 0, "negative": -1}
)

# Build per-aspect summary
summary = aspects.groupby("aspect").agg(
    total      = ("sentiment_score", "count"),
    mean_sent  = ("sentiment_score", "mean"),
    pos_pct    = ("sentiment_score", lambda x: (x == 1).mean() * 100),
    neg_pct    = ("sentiment_score", lambda x: (x == -1).mean() * 100),
).reset_index()

# Merge CI data
ci_df["aspect_key"] = ci_df["aspect"]
summary = summary.merge(
    ci_df[["aspect_key", "ci_lo", "ci_hi"]],
    left_on="aspect", right_on="aspect_key", how="left"
).drop(columns=["aspect_key"])

summary["label"] = summary["aspect"].str.replace("_", " ").str.title()

# Colour by signal
def get_color(row):
    if row["ci_lo"] > 0:   return "#34c759"  # green — statistically positive
    if row["ci_hi"] < 0:   return "#ff453a"  # red   — statistically negative
    return "#0a84ff"                          # blue  — mixed/uncertain

summary["color"] = summary.apply(get_color, axis=1)

# Bubble size scaled to total mentions
max_total = summary["total"].max()
summary["bubble_size"] = (summary["total"] / max_total * 60 + 15).round(1)

# Hover text
summary["hover"] = summary.apply(lambda r: (
    f"<b>{r['label']}</b><br>"
    f"Mentions: {int(r['total']):,}<br>"
    f"Mean sentiment: {r['mean_sent']:+.3f}<br>"
    f"95% CI: [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]<br>"
    f"Positive: {r['pos_pct']:.1f}%<br>"
    f"Negative: {r['neg_pct']:.1f}%"
), axis=1)

fig = go.Figure()

# Reference lines
fig.add_hline(y=0, line_dash="dot", line_color="#3a3a3c",
              annotation_text="Neutral", annotation_font_color="#6e6e73",
              annotation_position="right")

# Bubbles
fig.add_trace(go.Scatter(
    x=summary["total"],
    y=summary["mean_sent"],
    mode="markers+text",
    marker=dict(
        size=summary["bubble_size"],
        color=summary["color"],
        opacity=0.85,
        line=dict(color="#000", width=1),
        sizemode="diameter",
    ),
    text=summary["label"],
    textposition="top center",
    textfont=dict(size=11, color="#f5f5f7"),
    hovertext=summary["hover"],
    hoverinfo="text",
    name="",
))

# Quadrant annotations
fig.add_annotation(x=0.97, y=0.97, xref="paper", yref="paper",
    text="High discussion · Positive", showarrow=False,
    font=dict(size=10, color="#34c759"), xanchor="right")
fig.add_annotation(x=0.97, y=0.03, xref="paper", yref="paper",
    text="High discussion · Negative", showarrow=False,
    font=dict(size=10, color="#ff453a"), xanchor="right")

fig.update_layout(
    title=dict(
        text="Feature Perception Map — iPhone Duo & iPhone 18 Pro",
        font=dict(size=16, color="#f5f5f7"),
    ),
    paper_bgcolor="#111111",
    plot_bgcolor="#111111",
    font=dict(family="-apple-system, SF Pro Display, sans-serif", color="#ebebf5"),
    xaxis=dict(
        title="Discussion volume (total mentions)",
        gridcolor="#1d1d1f", zerolinecolor="#1d1d1f",
        tickfont=dict(color="#6e6e73"),
    ),
    yaxis=dict(
        title="Mean sentiment score (−1 negative → +1 positive)",
        gridcolor="#1d1d1f", zerolinecolor="#2a2a2a",
        tickfont=dict(color="#6e6e73"),
        range=[-0.35, 0.35],
    ),
    showlegend=False,
    height=580,
    margin=dict(l=60, r=40, t=60, b=60),
)

out = FIGURES / "feature_perception_map.html"
fig.write_html(out)
print(f"Saved → {out}")
print("Open it in your browser — this is the signature visualization.")
