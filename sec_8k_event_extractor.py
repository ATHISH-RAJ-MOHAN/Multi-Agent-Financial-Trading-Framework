#!/usr/bin/env python3
"""
SEC 8-K Event Extractor (EDGAR) — Ready-to-Run

What this does
--------------
- Maps tickers → CIKs (via SEC's public ticker list)
- Pulls recent 8-K / 8-K/A filings from the Submissions API
- Downloads each filing's full-text HTML
- Extracts 8-K Items (e.g., 2.02 Earnings, 1.01 Material Agreements, 5.02 Officer/Director changes)
- Emits a normalized events table (CSV + Parquet) for downstream Event-Detection pipelines

Usage
-----
$ pip install requests pandas pyarrow beautifulsoup4 python-dateutil tqdm
$ export SEC_USER_AGENT="Your Name your.email@example.com"
$ python sec_8k_event_extractor.py --tickers AAPL,MSFT,TSLA --days 30 --out events_8k

Notes
-----
- Respect SEC rate limits: ~10 requests/sec max; this script uses conservative sleeps.
- You MUST set the SEC_USER_AGENT environment variable with your email per SEC guidelines.
- For large backfills, consider chunking by date ranges and caching results on disk.
"""

import os
import re
import io
import json
import time
import math
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Iterable, Tuple
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from tqdm import tqdm

SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TMPL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
SEC_ARCHIVES_INDEX_TMPL = "https://www.sec.gov/Archives/edgar/data/{cik_str}/{acc_no_nodash}/index.json"
SEC_ARCHIVES_DOC_TMPL = "https://www.sec.gov/Archives/edgar/data/{cik_str}/{acc_no_nodash}/{doc_name}"

USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip()

def _session() -> requests.Session:
    if not USER_AGENT:
        raise RuntimeError("Please set SEC_USER_AGENT env var: 'Your Name your.email@example.com'")
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return s

def _sleep(sec: float = 0.2):
    time.sleep(sec)

def load_ticker_cik_map(session: requests.Session) -> pd.DataFrame:
    """Load SEC master Ticker→CIK mapping (official file)."""
    _sleep()
    r = session.get(SEC_TICKER_CIK_URL, timeout=30)
    r.raise_for_status()
    # The JSON is an object with numeric keys -> {cik_str, ticker, title}
    raw = r.json()
    rows = []
    for _, obj in raw.items():
        rows.append({"ticker": obj["ticker"].upper(), "cik_str": str(obj["cik_str"]), "title": obj["title"]})
    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df

def pad_cik(cik_str: str) -> str:
    return str(cik_str).zfill(10)

def strip_dashes(accession_no: str) -> str:
    return accession_no.replace("-", "")

@dataclass
class EventRecord:
    event_id: str
    source: str
    ticker: Optional[str]
    cik: str
    company_name: Optional[str]
    event_type: str              # '8k_item'
    subtype: Optional[str]       # like '2.02', '1.01', etc.
    event_datetime: Optional[str]
    timezone: str
    fiscal_period: Optional[str]
    fiscal_year: Optional[str]
    numeric_payload_json: Optional[str]
    text_payload: Optional[str]  # short excerpt/snippet
    doc_uri: str
    confidence: float
    ingest_ts: str

ITEM_PATTERN = re.compile(
    r'(?:Item|ITEM)\s+((?:\d+)\.(?:\d+))\s*[—\-–:]?\s*(.{0,120})',  # capture item code and short title
    re.IGNORECASE | re.DOTALL
)

HEADER_CLEAN = re.compile(r'\s+')

def find_items_from_html(html: str) -> List[Tuple[str, str]]:
    """
    Heuristic extraction of 8-K item numbers and short titles from filing HTML text.
    Returns list of (item_code, short_title).
    """
    # Convert HTML to visible text quickly
    soup = BeautifulSoup(html, "html.parser")
    # Pull text but keep reasonable size
    text = soup.get_text("\n")
    # Normalize spacing
    text = HEADER_CLEAN.sub(" ", text)
    items = []
    for m in ITEM_PATTERN.finditer(text):
        code = m.group(1).strip()
        title = m.group(2).strip()
        # Trim trailing junk
        title = re.sub(r'\s{2,}', ' ', title)
        # Avoid duplicates
        items.append((code, title))
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for code, title in items:
        if code not in seen:
            uniq.append((code, title))
            seen.add(code)
    return uniq

def get_submissions(session: requests.Session, cik_padded: str) -> dict:
    _sleep()
    url = SEC_SUBMISSIONS_URL_TMPL.format(cik_padded=cik_padded)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def list_recent_8k_filings(submissions_json: dict, days_back: int) -> List[dict]:
    """Filter the 'recent' filings for 8-K/8-KA within the date window."""
    df = pd.DataFrame(submissions_json.get("filings", {}).get("recent", {}))
    if df.empty:
        return []
    # Normalize columns we need
    df = df.rename(columns={
        "accessionNumber": "accession_no",
        "form": "form",
        "filingDate": "filing_date",
        "reportDate": "report_date",
        "primaryDocument": "primary_doc",
        "primaryDocDescription": "primary_desc",
    })
    window_start = datetime.now(timezone.utc) - timedelta(days=days_back)
    # Keep 8-K and 8-K/A only
    df = df[df["form"].str.upper().isin(["8-K", "8-K/A"])].copy()
    # Filter by filing_date within window
    df["filing_dt"] = pd.to_datetime(df["filing_date"], errors="coerce", utc=True)
    df = df[df["filing_dt"] >= window_start]
    # Order newest first
    df = df.sort_values("filing_dt", ascending=False).reset_index(drop=True)
    return df.to_dict(orient="records")

def fetch_filing_html(session: requests.Session, cik_str: str, accession_no: str, primary_doc: str) -> Tuple[str, str]:
    """
    Try to fetch the main document for the filing. Fallback to first HTML/text doc if needed.
    Returns (doc_url, html_text).
    """
    # First load filing index.json to see all documents
    acc_nodash = strip_dashes(accession_no)
    idx_url = SEC_ARCHIVES_INDEX_TMPL.format(cik_str=str(int(cik_str)), acc_no_nodash=acc_nodash)
    _sleep()
    idx_r = session.get(idx_url, timeout=30)
    idx_r.raise_for_status()
    idx = idx_r.json()

    # Prefer primary_doc when present; else, first text/html doc
    docs = idx.get("directory", {}).get("item", [])
    # Find candidate documents
    preferred = None
    other_html = []
    for d in docs:
        name = d.get("name", "")
        if not name:
            continue
        # Prioritize the primary_doc (if it exists in the index)
        if primary_doc and name.lower() == primary_doc.lower():
            preferred = name
            break
        # Otherwise collect HTML/htm/txt candidates
        if name.lower().endswith((".htm", ".html", ".txt")):
            other_html.append(name)

    target_doc = preferred or (other_html[0] if other_html else None)
    if not target_doc:
        raise RuntimeError("No HTML/TXT document found in filing index.")

    doc_url = SEC_ARCHIVES_DOC_TMPL.format(cik_str=str(int(cik_str)), acc_no_nodash=acc_nodash, doc_name=target_doc)
    _sleep()
    doc_r = session.get(doc_url, timeout=60)
    doc_r.raise_for_status()
    return doc_url, doc_r.text

def normalize_event_rows(company_name: str,
                         ticker: Optional[str],
                         cik_str: str,
                         filing_row: dict,
                         items: List[Tuple[str, str]]) -> List[EventRecord]:
    rows = []
    filing_date = filing_row.get("filing_date") or filing_row.get("filing_dt")
    filing_dt = None
    try:
        filing_dt = dateparser.parse(filing_date).astimezone(timezone.utc).isoformat()
    except Exception:
        filing_dt = None

    for code, title in (items or [("unknown", "Unknown Item")]):  # at least one row
        event_id = f"{cik_str}:{filing_row['accession_no']}:{code}"
        rows.append(EventRecord(
            event_id=event_id,
            source="sec_edgar",
            ticker=ticker,
            cik=str(int(cik_str)),
            company_name=company_name,
            event_type="8k_item",
            subtype=code if code != "unknown" else None,
            event_datetime=filing_dt,
            timezone="UTC",
            fiscal_period=None,
            fiscal_year=None,
            numeric_payload_json=None,
            text_payload=title[:280] if title else None,
            doc_uri=None,  # filled in by caller
            confidence=0.9 if code != "unknown" else 0.5,
            ingest_ts=datetime.now(timezone.utc).isoformat()
        ))
    return rows

def fetch_company_name(submissions_json: dict) -> str:
    return submissions_json.get("name", "") or submissions_json.get("entityType", "") or ""

def run(tickers: List[str], days: int, out_prefix: str):
    session = _session()
    # Map tickers -> CIKs
    print("Loading ticker→CIK map...")
    tdf = load_ticker_cik_map(session)
    tdf["ticker"] = tdf["ticker"].str.upper()
    ticker_map = {row["ticker"]: row["cik_str"] for _, row in tdf.iterrows()}

    all_events: List[EventRecord] = []

    for tk in tickers:
        tk = tk.upper().strip()
        cik_str = ticker_map.get(tk)
        if not cik_str:
            print(f"[WARN] No CIK found for {tk}. Skipping.")
            continue
        cik_pad = pad_cik(cik_str)
        print(f"Processing {tk} (CIK {cik_pad}) ...")

        subs = get_submissions(session, cik_pad)
        company_name = fetch_company_name(subs)
        filings = list_recent_8k_filings(subs, days_back=days)

        for frow in tqdm(filings, desc=f"{tk} 8-Ks", unit="filing"):
            try:
                doc_url, html_text = fetch_filing_html(session, cik_str, frow["accession_no"], frow.get("primary_doc"))
                items = find_items_from_html(html_text)
                event_rows = normalize_event_rows(company_name, tk, cik_str, frow, items)
                for er in event_rows:
                    er.doc_uri = doc_url
                all_events.extend(event_rows)
            except Exception as e:
                print(f"[WARN] Failed filing {frow.get('accession_no')}: {e}")
                continue

    if not all_events:
        print("No events extracted.")
        return

    # Build DataFrame
    df = pd.DataFrame([asdict(r) for r in all_events])
    # Sort by event time desc
    if "event_datetime" in df.columns:
        df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce", utc=True)
        df = df.sort_values(["event_datetime", "ticker", "subtype"], ascending=[False, True, True])
    # Save
    csv_path = f"{out_prefix}.csv"
    parquet_path = f"{out_prefix}.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"[WARN] Parquet write failed ({e}); CSV saved anyway.")
        parquet_path = None

    print(f"✔ Wrote {len(df)} events")
    print(f"CSV: {os.path.abspath(csv_path)}")
    if parquet_path:
        print(f"Parquet: {os.path.abspath(parquet_path)}")

def parse_args():
    ap = argparse.ArgumentParser(description="SEC 8-K Event Extractor (EDGAR)")
    ap.add_argument("--tickers", type=str, required=True, help="Comma-separated tickers, e.g., AAPL,MSFT,TSLA")
    ap.add_argument("--days", type=int, default=30, help="Look-back window in days (default: 30)")
    ap.add_argument("--out", type=str, default="events_8k", help="Output file prefix (default: events_8k)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    run(tickers, args.days, args.out)
