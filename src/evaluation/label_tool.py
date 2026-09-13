"""
Manual Labelling Tool
=====================
A simple Streamlit app to manually label comments for ground-truth evaluation.

Labels 200-300 comments with:
  - aspect (which feature is being discussed)
  - sentiment (positive / negative / neutral)
  - purchase_intent (buy / considering / not_buying / conditional / unknown)

Saves progress automatically — resume any time.

Run from apple-effect/ root:
    streamlit run src/evaluation/label_tool.py
"""

import json
import random
from pathlib import Path

import pandas as pd
import streamlit as st

PROC       = Path("data/processed")
LABEL_FILE = Path("data/processed/manual_labels.json")
TARGET     = 250   # how many to label

st.set_page_config(page_title="Label Tool · Apple Effect", page_icon="🏷️", layout="centered")

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, 'SF Pro Display', sans-serif !important;
    background: #000 !important; color: #f5f5f7 !important;
}
.main { background: #000 !important; }
.block-container { max-width: 780px !important; padding: 2rem !important; }
.comment-box {
    background: #111; border: 0.5px solid #2a2a2a; border-radius: 14px;
    padding: 22px 24px; font-size: 15px; line-height: 1.7;
    color: #f5f5f7; margin: 20px 0;
}
.progress-label { font-size: 11px; color: #6e6e73; letter-spacing: 0.5px; text-transform: uppercase; }
.stRadio > label { color: #6e6e73 !important; font-size: 12px !important; }
div[role="radiogroup"] label { color: #f5f5f7 !important; font-size: 13px !important; padding: 6px 10px !important; border-radius: 8px !important; }
div[role="radiogroup"] label:hover { background: #1a1a1a !important; }
[data-testid="stButton"] button {
    background: #0a84ff !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-size: 14px !important; font-weight: 500 !important;
    padding: 10px 28px !important; width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_comments():
    df = pd.read_csv(PROC / "comments_english.csv")
    # Sample a diverse set — stratified by comment length buckets
    df["wc"] = df["text_clean"].str.split().str.len()
    df = df[df["wc"].between(5, 150)].copy()
    sampled = df.sample(min(TARGET * 3, len(df)), random_state=42)
    return sampled.reset_index(drop=True)

try:
    comments = load_comments()
except FileNotFoundError:
    st.error("Run `python src/nlp/run_absa.py` first to generate comments_english.csv")
    st.stop()

# ── Load existing labels ───────────────────────────────────────────────────────
if LABEL_FILE.exists():
    labels = json.loads(LABEL_FILE.read_text())
else:
    labels = {}

labeled_ids = set(labels.keys())
unlabeled   = [i for i in range(len(comments)) if str(i) not in labeled_ids]

# Auto-skip non-English comments
def is_english(text):
    text = str(text)
    if len(text) == 0: return False
    return (sum(1 for c in text if ord(c) < 128) / len(text)) >= 0.85

unlabeled_en = [i for i in unlabeled if is_english(comments.iloc[i]["text_clean"])]

# ── Header ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 🏷️ Label Tool")
    st.markdown('<p style="color:#6e6e73;font-size:13px;margin-top:-8px">Apple Effect · Ground truth creation</p>', unsafe_allow_html=True)
with col2:
    done = len(labeled_ids)
    pct  = done / TARGET * 100
    st.markdown(f'<div style="text-align:right"><div style="font-size:28px;font-weight:600;color:#f5f5f7">{done}</div><div style="font-size:11px;color:#6e6e73">of {TARGET} labelled</div></div>', unsafe_allow_html=True)

# Progress bar
st.markdown(f"""
<div style="margin:8px 0 24px">
    <div style="height:4px;background:#1d1d1f;border-radius:2px">
        <div style="width:{min(pct,100):.1f}%;height:4px;background:#0a84ff;border-radius:2px"></div>
    </div>
    <div style="font-size:11px;color:#6e6e73;margin-top:6px">{pct:.1f}% complete</div>
</div>
""", unsafe_allow_html=True)

if not unlabeled_en:
    st.warning("No more English comments to label.")
    st.stop()

if done >= TARGET:
    st.success(f"✅ Labelling complete! {done} comments labelled. Run the evaluation script next.")
    if st.button("Download labels as CSV"):
        rows = []
        for idx, lbl in labels.items():
            row = comments.iloc[int(idx)].to_dict()
            row.update(lbl)
            rows.append(row)
        df_out = pd.DataFrame(rows)
        st.download_button("Download", df_out.to_csv(index=False),
                           "manual_labels.csv", "text/csv")
    st.stop()

if not unlabeled:
    st.warning("All sampled comments labelled. Increase TARGET to label more.")
    st.stop()

# ── Pick next comment ─────────────────────────────────────────────────────────
# Auto-skip non-English comments (>30% non-ASCII chars)
def is_english(text):
    text = str(text)
    if len(text) == 0: return False
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    return ascii_ratio >= 0.85

# Filter unlabeled to English only
unlabeled_en = [i for i in unlabeled if is_english(comments.iloc[i]["text_clean"])]

if "current_idx" not in st.session_state or st.session_state.current_idx not in unlabeled_en:
    st.session_state.current_idx = unlabeled_en[0] if unlabeled_en else None

idx     = st.session_state.current_idx
comment = comments.iloc[idx]

# ── Display comment ───────────────────────────────────────────────────────────
st.markdown(f'<div class="comment-box">"{comment["text_clean"]}"</div>', unsafe_allow_html=True)

# Metadata
st.markdown(
    f'<div style="font-size:11px;color:#3a3a3c;margin-bottom:20px">'
    f'Comment #{idx} · {int(comment.get("word_count", comment["text_clean"].split().__len__()))} words'
    f'</div>', unsafe_allow_html=True)

# ── Label inputs ──────────────────────────────────────────────────────────────
st.markdown('<div style="font-size:11px;font-weight:500;color:#6e6e73;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:10px">What feature is this comment mainly about?</div>', unsafe_allow_html=True)

aspect = st.radio("Aspect", [
    "foldable_design", "camera", "price", "battery", "display",
    "ai_siri", "performance", "durability", "ecosystem",
    "comparison_samsung", "upgrade", "multitasking", "none / general"
], horizontal=True, label_visibility="collapsed")

st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:11px;font-weight:500;color:#6e6e73;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:10px">What is the sentiment toward that feature?</div>', unsafe_allow_html=True)

sentiment = st.radio("Sentiment", ["positive", "neutral", "negative"],
                     horizontal=True, label_visibility="collapsed")

st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
st.markdown('<div style="font-size:11px;font-weight:500;color:#6e6e73;letter-spacing:0.8px;text-transform:uppercase;margin-bottom:10px">Does this comment express purchase intent?</div>', unsafe_allow_html=True)

intent = st.radio("Intent", ["buy", "considering", "not_buying", "conditional", "unknown"],
                  horizontal=True, label_visibility="collapsed")

st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

col_a, col_b = st.columns([3, 1])
with col_a:
    if st.button("Save & Next →"):
        labels[str(idx)] = {
            "aspect":    aspect,
            "sentiment": sentiment,
            "intent":    intent,
        }
        LABEL_FILE.write_text(json.dumps(labels, indent=2))
        # move to next English comment
        remaining = [i for i in unlabeled_en if str(i) not in labels]
        if remaining:
            st.session_state.current_idx = remaining[0]
        st.rerun()
with col_b:
    if st.button("Skip"):
        # Mark as skipped so it never comes back
        labels[str(idx)] = {"aspect": "SKIPPED", "sentiment": "SKIPPED", "intent": "SKIPPED"}
        LABEL_FILE.write_text(json.dumps(labels, indent=2))
        remaining = [i for i in unlabeled_en if str(i) not in labels and i != idx]
        if remaining:
            st.session_state.current_idx = remaining[0]
        st.rerun()

# ── Recent labels ─────────────────────────────────────────────────────────────
if len(labels) > 0:
    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#6e6e73;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:10px">Recent labels</div>', unsafe_allow_html=True)
    recent = list(labels.items())[-5:][::-1]
    for i, (lidx, lbl) in enumerate(recent):
        text = comments.iloc[int(lidx)]["text_clean"][:80]
        color = {"positive":"#34c759","negative":"#ff453a","neutral":"#6e6e73"}.get(lbl["sentiment"],"#6e6e73")
        st.markdown(
            f'<div style="font-size:12px;color:#6e6e73;padding:6px 0;border-top:0.5px solid #1d1d1f">'
            f'<span style="color:{color}">■</span> '
            f'<span style="color:#ebebf5">{lbl["aspect"]}</span> · '
            f'<span style="color:{color}">{lbl["sentiment"]}</span> · '
            f'{text}…</div>', unsafe_allow_html=True)
