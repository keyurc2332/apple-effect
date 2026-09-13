"""
YouTube Data Collector
======================
Collects videos and comments for Apple product launch analysis.

Design principles:
  - Discover → save IDs → batch-fetch details  (quota-efficient)
  - Never re-fetch what's already on disk
  - Respects the 10,000-unit/day default quota
  - Saves raw JSON + clean CSVs

Quota cost summary:
  search.list          100 units / call   (50 results per call)
  videos.list            1 unit  / call   (up to 50 IDs per call)
  commentThreads.list    1 unit  / call   (up to 100 comments per call)
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
YOUTUBE_KEY = os.getenv("YOUTUBE_API_KEY")
RAW_DIR     = Path("data/raw/youtube")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Queries focused on the iPhone Duo launch + Pro comparison
SEARCH_QUERIES = [
    "iPhone Duo review",
    "iPhone Duo foldable hands on",
    "iPhone 18 Pro review",
    "iPhone 18 Pro Max review",
    "iPhone Duo vs Galaxy Z Fold",
    "iPhone Duo price worth it",
    "iPhone Duo camera test",
    "Apple iPhone Duo unboxing",
    "iPhone 18 Pro camera test",
]

# Only videos published on/after launch announcement date
PUBLISHED_AFTER = "2026-09-01T00:00:00Z"

# Pilot mode: 5 videos/query  →  ~45 videos, ~5k comments
# Full mode:  50 videos/query →  400+ videos, ~50k comments
PILOT_MODE = False
RESULTS_PER_QUERY = 5 if PILOT_MODE else 50
MAX_COMMENTS_PER_VIDEO = 200


# ── Build service ────────────────────────────────────────────────────────────
def get_service():
    if not YOUTUBE_KEY:
        raise EnvironmentError("YOUTUBE_API_KEY not set in .env")
    return build("youtube", "v3", developerKey=YOUTUBE_KEY)


# ── Step 1: Discover video IDs ───────────────────────────────────────────────
def discover_video_ids(service) -> set[str]:
    """Search for relevant videos across all queries. Returns a deduplicated set of IDs."""
    all_ids: set[str] = set()
    id_file = RAW_DIR / "video_ids.json"

    # resume from disk if available
    if id_file.exists():
        existing = json.loads(id_file.read_text())
        log.info(f"Loaded {len(existing)} existing video IDs from disk")
        all_ids.update(existing)

    for query in SEARCH_QUERIES:
        log.info(f"Searching: {query!r}")
        try:
            page_token = None
            fetched_this_query = 0
            while fetched_this_query < RESULTS_PER_QUERY:
                resp = (
                    service.search()
                    .list(
                        q=query,
                        part="id,snippet",
                        type="video",
                        maxResults=min(50, RESULTS_PER_QUERY - fetched_this_query),
                        order="relevance",
                        publishedAfter=PUBLISHED_AFTER,
                        pageToken=page_token,
                    )
                    .execute()
                )
                items = resp.get("items", [])
                for item in items:
                    all_ids.add(item["id"]["videoId"])
                fetched_this_query += len(items)
                page_token = resp.get("nextPageToken")
                if not page_token or not items:
                    break
                time.sleep(0.5)  # be polite
        except HttpError as e:
            log.error(f"Search failed for {query!r}: {e}")
            if e.status_code == 403:
                log.error("Quota likely exhausted — stopping search phase")
                break

    id_file.write_text(json.dumps(list(all_ids)))
    log.info(f"Total unique video IDs discovered: {len(all_ids)}")
    return all_ids


# ── Step 2: Fetch video details in batches ───────────────────────────────────
def fetch_video_details(service, video_ids: set[str]) -> pd.DataFrame:
    """Fetch statistics + snippet for all IDs. Batches of 50 (1 quota unit each)."""
    details_file = RAW_DIR / "video_details.json"
    existing_details: dict = {}
    if details_file.exists():
        existing_details = {v["id"]: v for v in json.loads(details_file.read_text())}

    ids_to_fetch = [vid for vid in video_ids if vid not in existing_details]
    log.info(f"Fetching details for {len(ids_to_fetch)} new videos (already have {len(existing_details)})")

    batch_size = 50
    all_details = dict(existing_details)

    for i in range(0, len(ids_to_fetch), batch_size):
        batch = ids_to_fetch[i : i + batch_size]
        try:
            resp = (
                service.videos()
                .list(part="id,snippet,statistics,contentDetails", id=",".join(batch))
                .execute()
            )
            for item in resp.get("items", []):
                all_details[item["id"]] = item
            log.info(f"  Batch {i // batch_size + 1}: fetched {len(resp.get('items', []))} details")
            time.sleep(0.3)
        except HttpError as e:
            log.error(f"Video detail fetch failed: {e}")

    details_file.write_text(json.dumps(list(all_details.values()), default=str))

    # flatten to DataFrame
    rows = []
    for item in all_details.values():
        snippet = item.get("snippet", {})
        stats   = item.get("statistics", {})
        rows.append({
            "video_id":      item["id"],
            "title":         snippet.get("title", ""),
            "channel":       snippet.get("channelTitle", ""),
            "published_at":  snippet.get("publishedAt", ""),
            "description":   snippet.get("description", "")[:500],
            "view_count":    int(stats.get("viewCount",    0) or 0),
            "like_count":    int(stats.get("likeCount",    0) or 0),
            "comment_count": int(stats.get("commentCount", 0) or 0),
            "category_id":   snippet.get("categoryId", ""),
            "tags":          "|".join(snippet.get("tags", [])),
        })

    df = pd.DataFrame(rows)
    csv_path = RAW_DIR / "videos.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Videos saved → {csv_path}  ({len(df)} rows)")
    return df


# ── Step 3: Collect comments ─────────────────────────────────────────────────
def collect_comments(service, video_ids: set[str]) -> pd.DataFrame:
    """Collect top-level comments for each video. Skips videos already scraped."""
    comments_dir = RAW_DIR / "comments_by_video"
    comments_dir.mkdir(exist_ok=True)

    all_comments = []

    for vid in video_ids:
        out_file = comments_dir / f"{vid}.json"
        if out_file.exists():
            # already collected — load from cache
            cached = json.loads(out_file.read_text())
            all_comments.extend(cached)
            continue

        video_comments = []
        page_token = None
        try:
            while len(video_comments) < MAX_COMMENTS_PER_VIDEO:
                resp = (
                    service.commentThreads()
                    .list(
                        part="snippet",
                        videoId=vid,
                        maxResults=min(100, MAX_COMMENTS_PER_VIDEO - len(video_comments)),
                        order="relevance",
                        pageToken=page_token,
                    )
                    .execute()
                )
                for item in resp.get("items", []):
                    top = item["snippet"]["topLevelComment"]["snippet"]
                    video_comments.append({
                        "video_id":         vid,
                        "comment_id":       item["id"],
                        "text":             top.get("textDisplay", ""),
                        "like_count":       top.get("likeCount", 0),
                        "published_at":     top.get("publishedAt", ""),
                        "reply_count":      item["snippet"].get("totalReplyCount", 0),
                    })
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
                time.sleep(0.3)

            out_file.write_text(json.dumps(video_comments, default=str))
            log.info(f"  {vid}: {len(video_comments)} comments")
            all_comments.extend(video_comments)

        except HttpError as e:
            if e.status_code in (403, 404):
                log.warning(f"  {vid}: comments disabled or video unavailable — skipping")
            else:
                log.error(f"  {vid}: comment fetch error — {e}")
            # write empty so we don't retry
            out_file.write_text("[]")
        time.sleep(0.5)

    df = pd.DataFrame(all_comments)
    if not df.empty:
        csv_path = RAW_DIR / "comments.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Comments saved → {csv_path}  ({len(df)} rows)")
    return df


# ── Orchestrator ─────────────────────────────────────────────────────────────
def run_pilot():
    """Full Phase 1 pilot collection."""
    log.info("=== Apple Effect — Phase 1: Pilot Data Collection ===")
    log.info(f"Mode: {'PILOT' if PILOT_MODE else 'FULL'}  |  {RESULTS_PER_QUERY} results/query")

    service = get_service()

    video_ids = discover_video_ids(service)
    videos_df = fetch_video_details(service, video_ids)
    comments_df = collect_comments(service, video_ids)

    # Quick summary
    print("\n" + "─" * 50)
    print("COLLECTION SUMMARY")
    print(f"  Videos collected:   {len(videos_df)}")
    print(f"  Comments collected: {len(comments_df)}")
    if not comments_df.empty:
        print(f"  Avg comments/video: {len(comments_df) / max(len(videos_df), 1):.1f}")
        print(f"  Date range:         {comments_df['published_at'].min()[:10]} → {comments_df['published_at'].max()[:10]}")
    print(f"  Raw data saved →    {RAW_DIR}")
    print("─" * 50)

    return videos_df, comments_df


if __name__ == "__main__":
    run_pilot()
