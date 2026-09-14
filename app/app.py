"""
Apple Effect — Streamlit Dashboard (Cloud-optimised)
Only uses: pandas, plotly, streamlit, numpy, json
No heavy ML imports at startup.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROC    = Path("data/processed")
REPORTS = Path("outputs/reports")

st.set_page_config(
    page_title="Apple Effect",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] {
    font-family: -apple-system, 'SF Pro Display', 'Inter', sans-serif !important;
    background-color: #000000 !important; color: #f5f5f7 !important;
}
.main { background-color: #000000 !important; }
.block-container { padding: 2rem 2.5rem 2rem !important; max-width: 1200px !important; }
section[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    border-right: 0.5px solid #1d1d1f !important;
}
section[data-testid="stSidebar"] * { color: #f5f5f7 !important; }
div[role="radiogroup"] label {
    background: transparent !important; border: none !important;
    color: #6e6e73 !important; font-size: 13px !important;
    padding: 8px 12px !important; border-radius: 8px !important; cursor: pointer !important;
}
div[role="radiogroup"] label:hover { background: #1a1a1a !important; color: #f5f5f7 !important; }
h1 { font-size: 28px !important; font-weight: 600 !important; letter-spacing: -0.8px !important; color: #f5f5f7 !important; }
.ae-section-label { font-size:11px; font-weight:500; color:#6e6e73; letter-spacing:0.8px; text-transform:uppercase; margin-bottom:14px; margin-top:8px; }
.finding { border-left:2px solid #0a84ff; padding:10px 14px; margin-bottom:10px; background:rgba(10,132,255,0.05); border-radius:0 8px 8px 0; font-size:13px; color:#ebebf5; line-height:1.55; }
.ae-card { background:#111111; border:0.5px solid #2a2a2a; border-radius:16px; padding:24px; margin-bottom:16px; }
[data-testid="metric-container"] { background:#111111 !important; border:0.5px solid #2a2a2a !important; border-radius:14px !important; padding:18px 20px !important; }
[data-testid="metric-container"] label { color:#6e6e73 !important; font-size:12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#f5f5f7 !important; font-size:28px !important; font-weight:600 !important; letter-spacing:-0.8px !important; }
[data-testid="stDataFrame"] { border:0.5px solid #2a2a2a !important; border-radius:12px !important; }
[data-testid="stSelectbox"] > div > div { background:#1a1a1a !important; border:0.5px solid #3a3a3c !important; border-radius:10px !important; color:#f5f5f7 !important; }
hr { border-color:#1d1d1f !important; }
::-webkit-scrollbar { width:6px; } ::-webkit-scrollbar-track { background:#0a0a0a; } ::-webkit-scrollbar-thumb { background:#2a2a2a; border-radius:3px; }
</style>
""", unsafe_allow_html=True)


def apply_theme(fig, height=380, **kwargs):
    fig.update_layout(
        paper_bgcolor="#111111", plot_bgcolor="#111111",
        font=dict(family="-apple-system, SF Pro Display, Inter, sans-serif", color="#ebebf5"),
        margin=dict(l=16, r=16, t=36, b=16), height=height, **kwargs,
    )
    return fig


@st.cache_data
def load_data():
    d = {}
    for key, path in [
        ("aspects",  PROC / "absa_aspect_rows.csv"),
        ("comments", PROC / "absa_comment_level.csv"),
        ("videos",   Path("data/raw/youtube/videos.csv")),
        ("ci",       REPORTS / "aspect_confidence_intervals.csv"),
        ("models",   REPORTS / "model_comparison.csv"),
    ]:
        try:    d[key] = pd.read_csv(path)
        except: d[key] = pd.DataFrame()
    try:
        d["clustered"] = pd.read_csv(PROC / "comments_clustered.csv")
        d["umap"]      = np.load(PROC / "umap_2d.npy")
    except:
        d["clustered"] = None
        d["umap"]      = None
    try:
        d["honest_eval"] = json.loads((REPORTS / "honest_evaluation.json").read_text())
    except:
        d["honest_eval"] = None
    return d

data        = load_data()
aspects     = data["aspects"]
comments    = data["comments"]
videos      = data["videos"]
ci_df       = data["ci"]
model_df    = data["models"]
clustered   = data["clustered"]
honest_eval = data.get("honest_eval")

aspects_real = pd.DataFrame()
if len(aspects):
    aspects_real = aspects[aspects["aspect"] != "none"].copy()
    aspects_real["aspect_label"]    = aspects_real["aspect"].str.replace("_", " ").str.title()
    aspects_real["sentiment_score"] = aspects_real["sentiment"].map({"positive":1,"neutral":0,"negative":-1})


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 14px 0 24px">
        <div style="font-size: 11px; font-weight: 500; letter-spacing: 3px;
                    color: #6e6e73; text-transform: uppercase; margin-bottom: 6px">Product Intelligence</div>
        <div style="font-size: 22px; font-weight: 600; letter-spacing: -0.6px; color: #f5f5f7; line-height: 1.1">Apple<br>Effect</div>
        <div style="width: 28px; height: 2px; background: #0a84ff; border-radius: 1px; margin-top: 10px"></div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("nav",
        ["Overview","Feature Intelligence","Consumer Voice","Purchase Intent","ML Models"],
        label_visibility="collapsed")
    st.divider()
    if len(comments):
        st.markdown(
            f'<div style="font-size:12px;color:#6e6e73;line-height:2.2">'
            f'<span style="color:#f5f5f7;font-weight:500">{len(comments):,}</span> comments<br>'
            f'<span style="color:#f5f5f7;font-weight:500">{len(videos):,}</span> videos<br>'
            f'<span style="color:#f5f5f7;font-weight:500">{len(aspects_real):,}</span> aspect mentions<br>'
            f'<span style="color:#f5f5f7;font-weight:500">Sep 9–13, 2026</span></div>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div style="margin-bottom: 8px">
        <div style="font-size: 11px; font-weight: 500; letter-spacing: 3px; color: #6e6e73; text-transform: uppercase; margin-bottom: 8px">Product Intelligence</div>
        <div style="font-size: 38px; font-weight: 600; letter-spacing: -1.2px; color: #f5f5f7; line-height: 1.05">Apple Effect</div>
        <div style="font-size: 14px; color: #6e6e73; margin-top: 8px; letter-spacing: -0.2px">What consumers actually think about the iPhone Duo &amp; iPhone 18 Pro</div>
    </div>
    <div style="display:flex;gap:8px;margin:16px 0 28px">
        <span style="font-size:11px;padding:4px 12px;border-radius:20px;background:rgba(10,132,255,0.1);color:#0a84ff;border:0.5px solid rgba(10,132,255,0.3)">iPhone Duo</span>
        <span style="font-size:11px;padding:4px 12px;border-radius:20px;background:rgba(255,255,255,0.05);color:#8e8e93;border:0.5px solid #2a2a2a">iPhone 18 Pro</span>
        <span style="font-size:11px;padding:4px 12px;border-radius:20px;background:rgba(52,199,89,0.1);color:#34c759;border:0.5px solid rgba(52,199,89,0.3)">Launch week · Sep 2026</span>
    </div>
    <div style="height:1px;background:#1d1d1f;margin-bottom:28px"></div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Comments analysed", f"{len(comments):,}", delta="22,756 collected")
    with c2: st.metric("YouTube videos", f"{len(videos):,}", delta="Launch week")
    with c3: st.metric("Aspect mentions", f"{len(aspects_real):,}", delta="12 features")
    with c4: st.metric("Human-validated F1", "0.384", delta="κ=0.184 · 175 labels")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="ae-section-label">Discussion volume by feature</div>', unsafe_allow_html=True)
        if len(aspects_real):
            vol = aspects_real.groupby("aspect_label").size().sort_values(ascending=True).reset_index()
            vol.columns = ["Feature", "Mentions"]
            fig = px.bar(vol, x="Mentions", y="Feature", orientation="h",
                         color="Mentions", color_continuous_scale=["#0a3a6e","#0a84ff"],
                         template="plotly_dark")
            fig.update_traces(marker_line_width=0)
            apply_theme(fig, height=380, coloraxis_showscale=False,
                        xaxis=dict(gridcolor="#1d1d1f", zerolinecolor="#1d1d1f"),
                        yaxis=dict(gridcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="ae-section-label">Key findings</div>', unsafe_allow_html=True)
        for icon, title, body in [
            ("🔴","Durability","Only feature with a statistically negative CI — consumers fear the hinge."),
            ("🤖","AI / Siri","#1 topic (4,769 mentions) with the most unresolved sentiment."),
            ("💰","Price","Mixed, not negative — consumers debate value, not reject it."),
            ("✅","Camera · Battery","Strongest positive signals across all features."),
            ("📊","Honest F1 = 0.384","Human-validated benchmark, not circular auto-label score."),
        ]:
            st.markdown(f'<div class="finding"><strong>{icon} {title}</strong> — {body}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="ae-section-label">Top videos by views</div>', unsafe_allow_html=True)
    if len(videos):
        top = videos.sort_values("view_count", ascending=False).head(8)[["title","channel","view_count","comment_count"]].copy()
        top["view_count"]    = top["view_count"].apply(lambda x: f"{int(x):,}")
        top["comment_count"] = top["comment_count"].apply(lambda x: f"{int(x):,}")
        top.columns = ["Title","Channel","Views","Comments"]
        st.dataframe(top, use_container_width=True, hide_index=True,
                     column_config={"Title": st.column_config.TextColumn(width="large"),
                                    "Views": st.column_config.TextColumn(width="small")})


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FEATURE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Intelligence":
    st.markdown('<h1 style="margin-bottom:4px">Feature Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6e6e73;font-size:14px;margin-bottom:28px">Aspect-level sentiment with bootstrap 95% confidence intervals</p>', unsafe_allow_html=True)

    if len(aspects_real):
        pivot = (aspects_real.groupby(["aspect_label","sentiment"])
                 .size().unstack(fill_value=0)
                 .reindex(columns=["positive","neutral","negative"], fill_value=0))
        pivot["total"]   = pivot.sum(axis=1)
        pivot["pos_pct"] = (pivot.get("positive",0) / pivot["total"] * 100).round(1)
        pivot["neg_pct"] = (pivot.get("negative",0) / pivot["total"] * 100).round(1)
        pivot["net"]     = (pivot["pos_pct"] - pivot["neg_pct"]).round(1)
        pivot = pivot.sort_values("net", ascending=False).reset_index()

        st.markdown('<div class="ae-section-label">Sentiment matrix</div>', unsafe_allow_html=True)
        display = pivot[["aspect_label","total","pos_pct","neg_pct","net"]].copy()
        display.columns = ["Feature","Mentions","Positive %","Negative %","Net Score"]
        st.dataframe(display, use_container_width=True, hide_index=True,
                     column_config={
                         "Positive %": st.column_config.ProgressColumn(min_value=0, max_value=35, format="%.1f%%"),
                         "Negative %": st.column_config.ProgressColumn(min_value=0, max_value=35, format="%.1f%%"),
                     })

    # Feature Perception Map
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="ae-section-label">Feature perception map</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6e6e73;font-size:12px;margin-bottom:16px">x = discussion volume · y = mean sentiment · colour = statistical signal (green=positive CI, red=negative CI, blue=mixed)</p>', unsafe_allow_html=True)

    if len(aspects_real) and len(ci_df):
        summary = aspects_real.groupby("aspect_label").agg(
            total=("sentiment_score","count"),
            mean_sent=("sentiment_score","mean"),
            pos_pct=("sentiment_score", lambda x: (x==1).mean()*100),
            neg_pct=("sentiment_score", lambda x: (x==-1).mean()*100),
        ).reset_index()
        ci_merge = ci_df.copy()
        ci_merge["aspect_label"] = ci_merge["aspect"].str.replace("_"," ").str.title()
        summary = summary.merge(ci_merge[["aspect_label","ci_lo","ci_hi"]], on="aspect_label", how="left")
        def bubble_color(row):
            if row.get("ci_lo",0) > 0: return "#34c759"
            if row.get("ci_hi",0) < 0: return "#ff453a"
            return "#0a84ff"
        summary["color"] = summary.apply(bubble_color, axis=1)
        max_total = summary["total"].max()
        summary["bubble_size"] = (summary["total"] / max_total * 55 + 14).round(1)
        hover = summary.apply(lambda r: (
            f"<b>{r['aspect_label']}</b><br>Mentions: {int(r['total']):,}<br>"
            f"Mean sentiment: {r['mean_sent']:+.3f}<br>Positive: {r['pos_pct']:.1f}%  Negative: {r['neg_pct']:.1f}%"
        ), axis=1)
        fig_map = go.Figure()
        fig_map.add_hline(y=0, line_dash="dot", line_color="#3a3a3c", opacity=0.6)
        fig_map.add_trace(go.Scatter(
            x=summary["total"], y=summary["mean_sent"],
            mode="markers+text",
            marker=dict(size=summary["bubble_size"], color=summary["color"], opacity=0.85,
                        line=dict(color="#000",width=1), sizemode="diameter"),
            text=summary["aspect_label"], textposition="top center",
            textfont=dict(size=10, color="#f5f5f7"),
            hovertext=hover, hoverinfo="text", name="",
        ))
        apply_theme(fig_map, height=480, showlegend=False,
                    xaxis_title="Discussion volume (mentions)",
                    yaxis_title="Mean sentiment (−1 → +1)",
                    xaxis=dict(gridcolor="#1d1d1f", zerolinecolor="#1d1d1f"),
                    yaxis=dict(gridcolor="#1d1d1f", zerolinecolor="#2a2a2a", range=[-0.32,0.32]))
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="ae-section-label">Confidence intervals</div>', unsafe_allow_html=True)
        if len(ci_df):
            ci_plot = ci_df.copy()
            ci_plot["label"] = ci_plot["aspect"].str.replace("_"," ").str.title()
            ci_plot = ci_plot.sort_values("mean_sentiment")
            colors = ["#ff453a" if hi < 0 else ("#34c759" if lo > 0 else "#0a84ff")
                      for lo, hi in zip(ci_plot["ci_lo"], ci_plot["ci_hi"])]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ci_plot["mean_sentiment"], y=ci_plot["label"],
                mode="markers", marker=dict(size=9, color=colors),
                error_x=dict(type="data", symmetric=False,
                             array=ci_plot["ci_hi"]-ci_plot["mean_sentiment"],
                             arrayminus=ci_plot["mean_sentiment"]-ci_plot["ci_lo"],
                             color="#3a3a3c", thickness=1.5)
            ))
            fig.add_vline(x=0, line_dash="dot", line_color="#3a3a3c", opacity=0.8)
            apply_theme(fig, height=380, showlegend=False,
                        xaxis_title="Mean sentiment score",
                        xaxis=dict(gridcolor="#1d1d1f", zerolinecolor="#1d1d1f"),
                        yaxis=dict(gridcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="ae-section-label">Drill into a feature</div>', unsafe_allow_html=True)
        if len(aspects_real):
            sel = st.selectbox("Feature", sorted(aspects_real["aspect_label"].unique()), label_visibility="collapsed")
            feat_df = aspects_real[aspects_real["aspect_label"] == sel]
            ca, cb, cc = st.columns(3)
            ca.metric("Mentions", f"{len(feat_df):,}")
            cb.metric("Positive", f"{(feat_df['sentiment']=='positive').mean()*100:.1f}%")
            cc.metric("Negative", f"{(feat_df['sentiment']=='negative').mean()*100:.1f}%")
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            for _, row in feat_df[feat_df["sentiment"].isin(["positive","negative"])].head(5).iterrows():
                color = "#34c759" if row["sentiment"] == "positive" else "#ff453a"
                icon  = "✅" if row["sentiment"] == "positive" else "❌"
                st.markdown(f'<div class="finding" style="border-left-color:{color}">{icon} {str(row["text"])[:200]}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CONSUMER VOICE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Consumer Voice":
    st.markdown('<h1 style="margin-bottom:4px">Consumer Voice</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6e6e73;font-size:14px;margin-bottom:28px">Topic clusters discovered by sentence embeddings + HDBSCAN</p>', unsafe_allow_html=True)

    if clustered is not None:
        cluster_names = {
            -1:"Noise", 0:"Non-English (DE)", 1:"Screen Crease Concern",
            2:"Samsung Ad Reaction", 3:"Non-English (Hindi)", 4:"Apple Pencil Discussion",
            5:"General Praise", 6:"Price Shock Humor", 7:"Regional Tech",
            8:"Camera & Display", 9:"Pro Max Specs", 10:"iPhone 18 Pro",
            11:"Design & Animation", 12:"General Apple", 13:"India Pricing",
            14:"Weight & Thickness", 15:"Samsung vs Apple Foldable",
            16:"Display Comparison", 17:"iPad Mini vs Duo", 18:"Wait and See",
        }
        plot_df = clustered[clustered["cluster"] != -1].copy()
        plot_df["cluster_name"] = plot_df["cluster"].map(cluster_names).fillna(plot_df["cluster"].astype(str))
        plot_df["hover_text"]   = plot_df["text_clean"].str[:100] + "…"
        fig = px.scatter(plot_df, x="umap_x", y="umap_y", color="cluster_name",
                         hover_data={"hover_text":True,"cluster_name":True,"umap_x":False,"umap_y":False},
                         template="plotly_dark", opacity=0.65)
        fig.update_traces(marker=dict(size=4))
        apply_theme(fig, height=520,
                    title="Comment topic clusters — UMAP + HDBSCAN",
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
                    legend_title_text="Topic",
                    legend=dict(font=dict(size=11), itemsizing="constant"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Cluster data not available in this deployment.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="ae-section-label">Apple vs consumer emphasis</div>', unsafe_allow_html=True)
    apple_em = {"A20 Pro chip":9,"Apple Intelligence":8,"Camera system":8,
                "Foldable design":9,"Battery life":6,"Multitasking":7,"Apple Pencil":5}
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p style="font-size:11px;color:#0a84ff;letter-spacing:0.5px;margin-bottom:8px">WHAT APPLE ANNOUNCED</p>', unsafe_allow_html=True)
        apple_df = pd.DataFrame(list(apple_em.items()), columns=["Feature","Score"])
        fig_a = px.bar(apple_df.sort_values("Score"), x="Score", y="Feature", orientation="h",
                       color_discrete_sequence=["#0a84ff"], template="plotly_dark")
        fig_a.update_traces(marker_line_width=0)
        apply_theme(fig_a, height=280, showlegend=False,
                    xaxis=dict(gridcolor="#1d1d1f", showticklabels=False),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_a, use_container_width=True)
    with col_b:
        if len(aspects_real):
            st.markdown('<p style="font-size:11px;color:#34c759;letter-spacing:0.5px;margin-bottom:8px">WHAT CONSUMERS DISCUSSED</p>', unsafe_allow_html=True)
            con_df = aspects_real["aspect_label"].value_counts().head(7).reset_index()
            con_df.columns = ["Feature","Mentions"]
            fig_c = px.bar(con_df.sort_values("Mentions"), x="Mentions", y="Feature", orientation="h",
                           color_discrete_sequence=["#34c759"], template="plotly_dark")
            fig_c.update_traces(marker_line_width=0)
            apply_theme(fig_c, height=280, showlegend=False,
                        xaxis=dict(gridcolor="#1d1d1f", showticklabels=False),
                        yaxis=dict(gridcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_c, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PURCHASE INTENT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Purchase Intent":
    st.markdown('<h1 style="margin-bottom:4px">Purchase Intent</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6e6e73;font-size:14px;margin-bottom:28px">Which features are associated with buying decisions?</p>', unsafe_allow_html=True)

    if len(comments):
        intent_counts = comments["purchase_intent"].value_counts()
        total = len(comments)
        label_map = {"buy":"Buying","considering":"Considering","not_buying":"Not buying",
                     "conditional":"Conditional","unknown":"Unknown"}
        emoji_map = {"buy":"🟢","considering":"🟡","not_buying":"🔴","conditional":"🟠","unknown":"⚪"}

        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown('<div class="ae-section-label">Intent distribution</div>', unsafe_allow_html=True)
            values = [intent_counts.get(k,0) for k in ["buy","considering","not_buying","conditional","unknown"]]
            fig = go.Figure(go.Pie(
                labels=["Buying","Considering","Not buying","Conditional","Unknown"],
                values=values, hole=0.6,
                marker=dict(colors=["#34c759","#ffd60a","#ff453a","#ff9f0a","#2a2a2a"],
                            line=dict(color="#000",width=1)),
                textinfo="label+percent", textfont=dict(size=11,color="#f5f5f7"),
            ))
            apply_theme(fig, height=320, showlegend=False,
                        annotations=[dict(text=f"<b>{total:,}</b>", x=0.5, y=0.5,
                                          showarrow=False, font=dict(size=16,color="#f5f5f7"))])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<div class="ae-section-label">Counts</div>', unsafe_allow_html=True)
            for k, v in intent_counts.items():
                st.markdown(
                    f'<div class="finding" style="padding:8px 14px">'
                    f'{emoji_map.get(k,"⚪")} <strong>{label_map.get(k,k)}</strong>'
                    f'<span style="float:right;color:#6e6e73">{v:,}&nbsp;({v/total*100:.1f}%)</span></div>',
                    unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        col3, col4 = st.columns(2)

        def aspect_bar(col, intent_key, color, title):
            subset = comments[comments["purchase_intent"]==intent_key]["aspects_found"].dropna()
            if not len(subset): return
            asp = pd.Series("|".join(subset).split("|"))
            asp = asp[asp != ""].value_counts().head(6).reset_index()
            asp.columns = ["Feature","Count"]
            asp["Feature"] = asp["Feature"].str.replace("_"," ").str.title()
            fig = px.bar(asp, x="Count", y="Feature", orientation="h",
                         color_discrete_sequence=[color], template="plotly_dark")
            fig.update_traces(marker_line_width=0)
            apply_theme(fig, height=250, showlegend=False, title=title,
                        title_font_size=13, title_font_color="#6e6e73",
                        xaxis=dict(gridcolor="#1d1d1f", showticklabels=False),
                        yaxis=dict(gridcolor="rgba(0,0,0,0)"))
            col.plotly_chart(fig, use_container_width=True)

        aspect_bar(col3, "buy",        "#34c759", "Features in BUY comments")
        aspect_bar(col4, "not_buying", "#ff453a", "Features in NOT BUYING comments")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="ae-section-label">Browse by intent</div>', unsafe_allow_html=True)
        intent_sel = st.selectbox("Intent", ["buy","not_buying","considering","conditional"],
                                  format_func=lambda x: label_map.get(x,x), label_visibility="collapsed")
        c_map = {"buy":"#34c759","not_buying":"#ff453a","considering":"#ffd60a","conditional":"#ff9f0a"}
        for _, row in comments[comments["purchase_intent"]==intent_sel].head(6).iterrows():
            st.markdown(f'<div class="finding" style="border-left-color:{c_map.get(intent_sel,"#0a84ff")}">{str(row["text"])[:240]}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ML Models":
    st.markdown('<h1 style="margin-bottom:4px">ML Models</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6e6e73;font-size:14px;margin-bottom:28px">Sentiment classification: majority baseline → TF-IDF → sentence embeddings</p>', unsafe_allow_html=True)

    # Human-validated evaluation — headline
    st.markdown('<div class="ae-section-label">Human-validated evaluation — primary result</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6e6e73;font-size:12px;margin-bottom:16px">5-fold CV on 175 manually labelled comments. This is the number that matters — not the auto-label benchmark.</p>', unsafe_allow_html=True)

    if honest_eval:
        s = honest_eval.get("sentiment", {})
        i = honest_eval.get("intent", {})
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("Human labels", f"{honest_eval.get('n_human_labels',175):,}")
        hc2.metric("Sentiment F1", f"{s.get('tfidf_lr_f1',0.384):.3f}", delta="vs baseline 0.190")
        hc3.metric("Cohen's κ", f"{s.get('tfidf_lr_kappa',0.184):.3f}", delta="slight-fair agreement")
        hc4.metric("Intent F1", f"{i.get('tfidf_lr_f1',0.327):.3f}" if i.get('tfidf_lr_f1') else "0.327")
        st.markdown("""
        <div class="finding" style="margin-top:12px">
            <strong>Why F1=0.384 is the right number to report.</strong>
            The auto-label experiment (F1=0.959) trains a model on its own generated labels —
            circular by design. The human-validated experiment shows genuine performance against
            175 independently labelled comments. Neutral is the hardest class: sarcasm, negation,
            and mixed opinions cause most misclassifications. This is an honest benchmark.
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="ae-section-label">Auto-label benchmark (pipeline consistency check)</div>', unsafe_allow_html=True)

    if len(model_df):
        color_map = {"Majority Baseline":"#3a3a3c",
                     "TF-IDF + Logistic Regression":"#34c759",
                     "Sentence Embeddings + Logistic Regression":"#ff9f0a"}
        for _, row in model_df.iterrows():
            color     = color_map.get(row["model"], "#0a84ff")
            pct       = row["macro_f1"] * 100
            delta     = row.get("improvement_over_baseline", 0)
            delta_str = f"+{delta:.3f} vs baseline" if delta > 0 else "baseline"
            st.markdown(f"""
            <div class="ae-card" style="display:flex;align-items:center;justify-content:space-between;padding:20px 24px;margin-bottom:10px">
                <div>
                    <div style="font-size:14px;font-weight:500;color:#f5f5f7">{row['model']}</div>
                    <div style="font-size:11px;color:#6e6e73;margin-top:4px">{delta_str}</div>
                    <div style="width:200px;height:3px;background:#1d1d1f;border-radius:2px;margin-top:12px">
                        <div style="width:{pct}%;height:3px;border-radius:2px;background:{color}"></div>
                    </div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:36px;font-weight:600;letter-spacing:-1.5px;color:{color};line-height:1">{row['macro_f1']:.3f}</div>
                    <div style="font-size:11px;color:#6e6e73;margin-top:4px">Macro F1</div>
                    <div style="font-size:11px;color:#6e6e73">Accuracy: {row['accuracy']:.3f}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="ae-section-label">Most predictive words</div>', unsafe_allow_html=True)
        for label, words, color, bg in [
            ("POSITIVE SIGNAL", ["love","buy","best","great","worth","amazing","awesome","beautiful"],
             "#34c759", "rgba(52,199,89,0.1)"),
            ("NEGATIVE SIGNAL", ["expensive","bad","hate","ugly","break","deal breaker","terrible","overpriced"],
             "#ff453a", "rgba(255,69,58,0.1)"),
        ]:
            st.markdown(f'<p style="font-size:11px;color:{color};letter-spacing:0.5px;margin:14px 0 8px">{label}</p>', unsafe_allow_html=True)
            chips = " ".join([f'<span style="display:inline-block;margin:3px;font-size:12px;padding:4px 11px;border-radius:20px;background:{bg};color:{color};border:0.5px solid {color}55">{w}</span>' for w in words])
            st.markdown(chips, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="ae-section-label">Pipeline architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="finding">Raw comments → text cleaning → English filter</div>
        <div class="finding">Rule-based aspect extraction (12 features)</div>
        <div class="finding">Lexicon sentiment → ML classification (TF-IDF + LR)</div>
        <div class="finding">Regex purchase intent detection</div>
        <div class="finding">Sentence embeddings → UMAP → HDBSCAN clusters</div>
        <div class="finding">Bootstrap CIs + chi-square statistical tests</div>
        <div class="finding">Human validation: 175 labels · 5-fold CV · F1=0.384</div>
        """, unsafe_allow_html=True)
