#!/usr/bin/env python3
# scripts/batch_run.py

"""
Batch-run the full AVAX Evolve pipeline multiple times and aggregate performance metrics.
"""
import subprocess
import sqlite3
import pandas as pd
import argparse
from pathlib import Path

# Path to results DB
DB_PATH = Path(__file__).parent.parent / "results.db"


def run_pipeline():
    """Invoke the main pipeline script."""
    subprocess.run(["./scripts/main.py"], check=True)


def fetch_recent_results(n: int, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load the last `n` strategy_runs from the SQLite database."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    conn = sqlite3.connect(str(db_path))
    query = """
        SELECT id, returns, sharpe, volatility, max_drawdown, timestamp
        FROM strategy_runs
        ORDER BY id DESC LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(n,), parse_dates=["timestamp"])
    conn.close()
    # Return oldest-first
    return df.iloc[::-1].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run AVAX Evolve and summarize results"
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of pipeline runs to execute"
    )
    args = parser.parse_args()

    print(f"🔁 Starting batch of {args.runs} runs...")
    for i in range(1, args.runs + 1):
        print(f"--- Run {i}/{args.runs} ---")
        run_pipeline()

    print(f"\n📊 Aggregating results from the last {args.runs} runs...")
    df = fetch_recent_results(args.runs)

    print("\nSummary Statistics:")
    print(df[['returns','sharpe','max_drawdown']].describe().to_string())

    print("\nIndividual Run Results:")
    print(df[['id','timestamp','returns','sharpe','max_drawdown']].to_string(index=False))

if __name__ == "__main__":
    main()
