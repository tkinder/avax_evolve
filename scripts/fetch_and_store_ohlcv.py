#!/usr/bin/env python3
# scripts/fetch_and_store_ohlcv.py

import sqlite3
import requests
import time
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent.parent / "results.db"
PRODUCT_ID = "AVAX-USD"
GRANULARITY = 86400  # daily
LIMIT = 300           # max per Coinbase request


def fetch_ohlcv_batch(start: datetime, end: datetime, product_id=PRODUCT_ID, granularity=GRANULARITY):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {"start": start.isoformat(), "end": end.isoformat(), "granularity": granularity}
    response = requests.get(url, params=params)
    response.raise_for_status()
    candles = response.json()
    return [
        {
            "timestamp": datetime.fromtimestamp(c[0], timezone.utc),
            "open": c[3],
            "high": c[2],
            "low": c[1],
            "close": c[4],
            "volume": c[5]
        }
        for c in candles
    ]


def get_latest_timestamp():
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Check if table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_data';")
    if not cur.fetchone():
        conn.close()
        return None
    cur.execute("SELECT MAX(timestamp) FROM ohlcv_data")
    row = cur.fetchone()[0]
    conn.close()
    return datetime.fromisoformat(row) if row else None


def store_ohlcv(records):
    if not records:
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS ohlcv_data (
            timestamp TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )''')
    for r in records:
        cur.execute(
            '''INSERT OR REPLACE INTO ohlcv_data (timestamp, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (r['timestamp'].isoformat(), r['open'], r['high'], r['low'], r['close'], r['volume'])
        )
    conn.commit()
    conn.close()


def fetch_full_history(days_back=1095):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    while start < end:
        batch_end = min(start + timedelta(seconds=GRANULARITY * LIMIT), end)
        print(f"🔄 Fetching: {start.date()} → {batch_end.date()}")
        data = fetch_ohlcv_batch(start, batch_end)
        store_ohlcv(data)
        start = batch_end
        time.sleep(1.1)
    print("✅ Full-history fetch complete.")


def fetch_new_data():
    latest = get_latest_timestamp()
    if not latest:
        print("⚠️ No existing data/table found. Please run with --full first.")
        return
    start = latest + timedelta(seconds=GRANULARITY)
    end = datetime.now(timezone.utc)
    if start >= end:
        print("✅ No new data to fetch.")
        return
    print(f"🔄 Fetching new data: {start.date()} → {end.date()}")
    data = fetch_ohlcv_batch(start, end)
    store_ohlcv(data)
    print(f"✅ Stored {len(data)} new rows.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch OHLCV data from Coinbase')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true', help='Fetch full historical data')
    group.add_argument('--new', action='store_true', help='Fetch only new data')
    args = parser.parse_args()

    if args.full:
        fetch_full_history()
    else:
        fetch_new_data()
