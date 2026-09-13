"""
Google Trends Collector
=======================
Pulls interest-over-time data for Apple launch queries.
Used as contextual evidence, not the central signal.
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

log = logging.getLogger(__name__)
RAW_DIR = Path("data/raw/trends")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Query groups (pytrends allows max 5 per payload)
QUERY_GROUPS = [
    ["iPhone Duo", "iPhone 18 Pro", "foldable iPhone", "iPhone 18"],
    ["iPhone Duo price", "iPhone Duo camera", "iPhone Duo battery", "iPhone Duo review"],
]

TIMEFRAME = "now 30-d"   # last 30 days to capture launch window


def collect_trends() -> dict[str, pd.DataFrame]:
    pt = TrendReq(hl="en-US", tz=360)
    results: dict[str, pd.DataFrame] = {}

    for i, queries in enumerate(QUERY_GROUPS):
        cache_path = RAW_DIR / f"group_{i}.csv"
        if cache_path.exists():
            log.info(f"Loading cached trends group {i}")
            results[f"group_{i}"] = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            continue

        log.info(f"Fetching trends group {i}: {queries}")
        try:
            pt.build_payload(queries, timeframe=TIMEFRAME, geo="", gprop="")
            df = pt.interest_over_time()
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            df.to_csv(cache_path)
            results[f"group_{i}"] = df
            log.info(f"  {len(df)} data points, columns: {list(df.columns)}")
        except Exception as e:
            log.error(f"  Trends group {i} failed: {e}")
        time.sleep(3)   # avoid rate limiting

    # also get related queries for the main term
    related_cache = RAW_DIR / "related_queries.json"
    if not related_cache.exists():
        try:
            pt.build_payload(["iPhone Duo"], timeframe=TIMEFRAME)
            related = pt.related_queries()
            related_cache.write_text(json.dumps(
                {k: {q_type: df.to_dict() if df is not None else None
                     for q_type, df in v.items()}
                 for k, v in related.items()},
                default=str
            ))
            log.info("Related queries saved")
        except Exception as e:
            log.warning(f"Related queries failed: {e}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dfs = collect_trends()
    for name, df in dfs.items():
        print(f"\n{name}:\n{df.tail(5)}")
