"""
Phase 0 — Data Source Validation
=================================
Run this before anything else.
Checks that each data source is reachable and returns meaningful data
for the Apple product launch queries.

Usage:
    python src/data_collection/validate_sources.py
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── colour helpers ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg):  print(f"  {CYAN}→{RESET} {msg}")
def header(msg):print(f"\n{BOLD}{msg}{RESET}")

results = {}  # source → {"status": ok|warn|fail, "notes": []}

# ── 1. YouTube Data API ──────────────────────────────────────────────────────
header("1 / 4  YouTube Data API")

YOUTUBE_KEY = os.getenv("YOUTUBE_API_KEY", "")

if not YOUTUBE_KEY:
    fail("YOUTUBE_API_KEY not set in .env")
    results["youtube"] = {"status": "fail", "notes": ["No API key"]}
else:
    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError

        yt = build("youtube", "v3", developerKey=YOUTUBE_KEY)

        # -- search for pilot query
        search_resp = (
            yt.search()
            .list(
                q="iPhone Duo foldable Apple",
                part="id,snippet",
                type="video",
                maxResults=5,
                order="relevance",
                publishedAfter="2026-09-01T00:00:00Z",
            )
            .execute()
        )
        items = search_resp.get("items", [])

        if not items:
            warn("Search returned 0 results — quota may be exhausted or query date too narrow")
            results["youtube"] = {"status": "warn", "notes": ["0 search results"]}
        else:
            ok(f"Search returned {len(items)} videos")
            for v in items[:3]:
                title = v["snippet"]["title"][:70]
                vid   = v["id"]["videoId"]
                info(f"{vid}  {title}")

            # -- fetch video details to confirm quota
            ids = [v["id"]["videoId"] for v in items]
            detail_resp = (
                yt.videos()
                .list(part="statistics,snippet", id=",".join(ids))
                .execute()
            )
            detail_items = detail_resp.get("items", [])
            ok(f"Video details fetched for {len(detail_items)} videos")

            # -- sample comment thread
            try:
                comment_resp = (
                    yt.commentThreads()
                    .list(
                        part="snippet",
                        videoId=ids[0],
                        maxResults=5,
                        order="relevance",
                    )
                    .execute()
                )
                comment_items = comment_resp.get("items", [])
                ok(f"Comment API accessible — {len(comment_items)} sample comments retrieved")
                for c in comment_items[:2]:
                    text = c["snippet"]["topLevelComment"]["snippet"]["textDisplay"][:80]
                    info(f'"{text}"')
                results["youtube"] = {"status": "ok", "notes": [
                    f"{len(items)} search results",
                    f"{len(comment_items)} sample comments"
                ]}
            except HttpError as e:
                warn(f"Comments disabled on that video (code {e.status_code}) — will skip per-video")
                results["youtube"] = {"status": "ok", "notes": ["Search/details ok", "Comments may be disabled on some videos"]}

    except ImportError:
        fail("google-api-python-client not installed. Run: pip install google-api-python-client")
        results["youtube"] = {"status": "fail", "notes": ["Library missing"]}
    except Exception as e:
        fail(f"YouTube API error: {e}")
        results["youtube"] = {"status": "fail", "notes": [str(e)]}


# ── 2. Reddit API ────────────────────────────────────────────────────────────
header("2 / 4  Reddit API")

REDDIT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_AGENT  = os.getenv("REDDIT_USER_AGENT", "apple-effect-research")

if not REDDIT_ID or not REDDIT_SECRET:
    warn("Reddit credentials not set — Reddit is OPTIONAL per project spec")
    info("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env when you have authorized access")
    results["reddit"] = {"status": "warn", "notes": ["No credentials — optional source"]}
else:
    try:
        import praw
        reddit = praw.Reddit(
            client_id=REDDIT_ID,
            client_secret=REDDIT_SECRET,
            user_agent=REDDIT_AGENT,
        )
        # light test — fetch 3 posts from r/apple
        posts = list(reddit.subreddit("apple").search("iPhone Duo", limit=3, sort="new"))
        if posts:
            ok(f"Reddit API accessible — {len(posts)} posts found in r/apple")
            for p in posts:
                info(f"{p.score:>6}↑  {p.title[:60]}")
            results["reddit"] = {"status": "ok", "notes": [f"{len(posts)} sample posts"]}
        else:
            warn("Reddit API connected but 0 posts returned")
            results["reddit"] = {"status": "warn", "notes": ["0 posts returned"]}
    except ImportError:
        warn("praw not installed — install only after you have authorized Reddit API access")
        results["reddit"] = {"status": "warn", "notes": ["praw not installed"]}
    except Exception as e:
        fail(f"Reddit API error: {e}")
        results["reddit"] = {"status": "fail", "notes": [str(e)]}


# ── 3. Google Trends ─────────────────────────────────────────────────────────
header("3 / 4  Google Trends (pytrends)")

try:
    from pytrends.request import TrendReq

    pt = TrendReq(hl="en-US", tz=360)
    queries = ["iPhone Duo", "iPhone 18 Pro", "foldable iPhone"]
    pt.build_payload(queries, cat=0, timeframe="now 7-d", geo="", gprop="")
    df = pt.interest_over_time()

    if df.empty:
        warn("Google Trends returned empty dataframe — possibly rate-limited, try again later")
        results["trends"] = {"status": "warn", "notes": ["Empty response — rate limit?"]}
    else:
        ok(f"Google Trends accessible — {len(df)} data points, {len(df.columns)} series")
        info(f"Columns: {list(df.columns)}")
        info(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
        results["trends"] = {"status": "ok", "notes": [
            f"{len(df)} data points",
            f"Queries: {queries}"
        ]}
except ImportError:
    fail("pytrends not installed. Run: pip install pytrends")
    results["trends"] = {"status": "fail", "notes": ["Library missing"]}
except Exception as e:
    fail(f"Google Trends error: {e}")
    results["trends"] = {"status": "warn", "notes": [str(e)]}


# ── 4. NLP libraries ─────────────────────────────────────────────────────────
header("4 / 4  NLP / ML libraries")

lib_checks = [
    ("transformers",          "Hugging Face Transformers"),
    ("sentence_transformers", "Sentence Transformers"),
    ("spacy",                 "spaCy"),
    ("sklearn",               "scikit-learn"),
    ("umap",                  "UMAP"),
    ("hdbscan",               "HDBSCAN"),
    ("xgboost",               "XGBoost"),
    ("plotly",                "Plotly"),
    ("streamlit",             "Streamlit"),
]

lib_ok, lib_fail = [], []
for module, label in lib_checks:
    try:
        __import__(module)
        ok(label)
        lib_ok.append(label)
    except ImportError:
        fail(f"{label}  (pip install {module.replace('_', '-')})")
        lib_fail.append(label)

results["libraries"] = {
    "status": "ok" if not lib_fail else ("warn" if lib_ok else "fail"),
    "notes": [f"Missing: {lib_fail}"] if lib_fail else ["All present"]
}


# ── Summary ──────────────────────────────────────────────────────────────────
header("─" * 50)
print(f"{BOLD}PHASE 0 SUMMARY{RESET}\n")

status_icons = {"ok": f"{GREEN}✓ READY{RESET}", "warn": f"{YELLOW}⚠ PARTIAL{RESET}", "fail": f"{RED}✗ BLOCKED{RESET}"}

for source, data in results.items():
    icon = status_icons.get(data["status"], "?")
    label = source.upper().ljust(12)
    notes = " | ".join(data["notes"])
    print(f"  {label} {icon}   {notes}")

print()

# decide whether to proceed
blocked = [s for s, d in results.items() if d["status"] == "fail" and s != "libraries"]
if "youtube" in blocked:
    print(f"{RED}YouTube is required. Fix the API key before continuing to Phase 1.{RESET}")
    sys.exit(1)
else:
    print(f"{GREEN}✓ YouTube is accessible — you can proceed to Phase 1 (pilot data collection).{RESET}")
    if results.get("reddit", {}).get("status") != "ok":
        warn("Reddit unavailable — project continues with YouTube + Trends as primary sources")

# save results
out = Path("outputs/reports/phase0_validation.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump({"run_at": datetime.utcnow().isoformat(), "results": results}, f, indent=2)
info(f"Results saved → {out}")
