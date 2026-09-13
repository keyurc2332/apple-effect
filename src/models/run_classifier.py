"""
Phase 5 — ML Classifier
========================
Trains and evaluates three models to classify comment sentiment per aspect.

Progression:
  1. Majority baseline
  2. TF-IDF + Logistic Regression
  3. Sentence Embedding + Logistic Regression  (our "transformer" approach)

We use the ABSA output as training signal:
  - Filter to comments with exactly one aspect (cleaner label)
  - Label = sentiment (positive / negative / neutral)
  - Feature = comment text

Run from apple-effect/ root:
    python src/models/run_classifier.py
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from sklearn.preprocessing import LabelEncoder
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

PROC    = Path("data/processed")
MODELS  = Path("models")
REPORTS = Path("outputs/reports")
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

SEP = "─" * 60

def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ── 1. Prepare dataset ────────────────────────────────────────────────────────
section("1 / 5  PREPARE TRAINING DATA")

df = pd.read_csv(PROC / "absa_aspect_rows.csv")

# Drop noise, non-aspect rows, non-English clusters
df = df[df["aspect"] != "none"]

# Keep only clear sentiment labels (drop neutral — too noisy for baseline)
df = df[df["sentiment"].isin(["positive", "negative"])]

print(f"  Total aspect-sentiment pairs: {len(df):,}")
print(f"\n  Sentiment distribution:")
print(df["sentiment"].value_counts().to_string())
print(f"\n  Aspect distribution (top 8):")
print(df["aspect"].value_counts().head(8).to_string())

# Features and labels
X_text = df["text"].fillna("").tolist()
y      = df["sentiment"].tolist()

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\n  Classes: {list(le.classes_)}")
print(f"  Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\n  Train: {len(X_train_text):,}  |  Test: {len(X_test_text):,}")


# ── Helper ────────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred, classes):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"\n  {name}")
    print(f"  Accuracy: {acc:.3f}   Macro F1: {f1:.3f}")
    print()
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix (rows=actual, cols=predicted):")
    header = "        " + "  ".join(f"{c:>10}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {classes[i]:>8}  " + "  ".join(f"{v:>10}" for v in row))
    return {"model": name, "accuracy": round(acc, 3), "macro_f1": round(f1, 3)}


results = []


# ── 2. Baseline — majority class ──────────────────────────────────────────────
section("2 / 5  BASELINE — MAJORITY CLASS")

majority = np.bincount(y_train).argmax()
y_pred_baseline = np.full(len(y_test), majority)
r = evaluate("Majority Baseline", y_test, y_pred_baseline, list(le.classes_))
results.append(r)


# ── 3. TF-IDF + Logistic Regression ──────────────────────────────────────────
section("3 / 5  TF-IDF + LOGISTIC REGRESSION")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words="english",
)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf  = tfidf.transform(X_test_text)

lr_tfidf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
lr_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = lr_tfidf.predict(X_test_tfidf)

r = evaluate("TF-IDF + Logistic Regression", y_test, y_pred_tfidf, list(le.classes_))
results.append(r)

# Save model
joblib.dump({"vectorizer": tfidf, "model": lr_tfidf, "encoder": le},
            MODELS / "tfidf_lr.pkl")
print(f"\n  Model saved → models/tfidf_lr.pkl")

# Top positive / negative features
print(f"\n  Most predictive words:")
feat_names = tfidf.get_feature_names_out()
coef = lr_tfidf.coef_[0]  # positive class coefficients
top_pos = np.argsort(coef)[-10:][::-1]
top_neg = np.argsort(coef)[:10]
print(f"  Positive signal: {', '.join(feat_names[i] for i in top_pos)}")
print(f"  Negative signal: {', '.join(feat_names[i] for i in top_neg)}")


# ── 4. Sentence Embeddings + Logistic Regression ──────────────────────────────
section("4 / 5  SENTENCE EMBEDDINGS + LOGISTIC REGRESSION")

EMB_CACHE = PROC / "embeddings.npy"
if not EMB_CACHE.exists():
    log.error("embeddings.npy not found — run src/nlp/run_embeddings.py first")
    sys.exit(1)

all_embeddings = np.load(EMB_CACHE)

# Match embeddings back to the filtered df rows
# The embeddings were built from comments_english.csv (4470 rows, index 0..4469)
# absa_aspect_rows has comment_idx referencing that index
valid_idx = df["comment_idx"].values
valid_idx = np.clip(valid_idx, 0, len(all_embeddings) - 1)
X_emb = all_embeddings[valid_idx]

X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(
    X_emb, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

lr_emb = LogisticRegression(max_iter=500, C=1.0, random_state=42)
lr_emb.fit(X_train_emb, y_train_emb)
y_pred_emb = lr_emb.predict(X_test_emb)

r = evaluate("Sentence Embeddings + Logistic Regression", y_test_emb, y_pred_emb, list(le.classes_))
results.append(r)

joblib.dump({"model": lr_emb, "encoder": le}, MODELS / "emb_lr.pkl")
print(f"\n  Model saved → models/emb_lr.pkl")


# ── 5. Summary table ──────────────────────────────────────────────────────────
section("5 / 5  MODEL COMPARISON")

summary_df = pd.DataFrame(results)
summary_df["improvement_over_baseline"] = (
    summary_df["macro_f1"] - summary_df.loc[0, "macro_f1"]
).round(3)

print(f"\n  {'Model':<45} {'Accuracy':>9} {'Macro F1':>9} {'vs Baseline':>12}")
print(f"  {'─'*45} {'─'*9} {'─'*9} {'─'*12}")
for _, row in summary_df.iterrows():
    delta = f"+{row['improvement_over_baseline']:.3f}" if row['improvement_over_baseline'] > 0 else f"{row['improvement_over_baseline']:.3f}"
    print(f"  {row['model']:<45} {row['accuracy']:>9.3f} {row['macro_f1']:>9.3f} {delta:>12}")

summary_df.to_csv(REPORTS / "model_comparison.csv", index=False)
print(f"\n  Saved → outputs/reports/model_comparison.csv")

best = summary_df.loc[summary_df["macro_f1"].idxmax(), "model"]
print(f"\n  Best model: {best}")

print(f"\n{SEP}")
print("PHASE 5 COMPLETE")
print("  Next: python src/models/run_statistical_analysis.py")
print(SEP)
