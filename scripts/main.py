#!/usr/bin/env python3
# scripts/main.py

import sys
from pathlib import Path
# Ensure project root is on PYTHONPATH so 'src' can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import subprocess
import sqlite3
from src.core.data import fetch_historical_data
from src.core.logging import get_logger
from src.evolution.phase1 import run_phase1_optimization
from src.evolution.phase2 import run_phase2_optimization
from src.core.results_logger import log_results

# Bootstrap OHLCV data
DB_PATH = Path(__file__).parent.parent / "results.db"
conn = sqlite3.connect(DB_PATH) if DB_PATH.exists() else None
has_ohlcv = False
if conn:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_data';")
    has_ohlcv = cur.fetchone() is not None
    conn.close()

# Fetch full history on first run or incremental update thereafter
if not has_ohlcv:
    print("🗄️  No OHLCV table found, running full fetch...")
    subprocess.run(["./scripts/fetch_and_store_ohlcv.py", "--full"], check=True)
else:
    print("🔄 Running incremental fetch...")
    subprocess.run(["./scripts/fetch_and_store_ohlcv.py", "--new"], check=True)

# Initialize logger
log = get_logger()

# Single pipeline execution
def main():
    log.info("🔥 Starting pipeline run")
    # Load data
    df = fetch_historical_data(refresh=False)
    log.info(f"📊 Loaded {len(df)} rows of historical data")

    # Phase 1
    log.info("=== Phase 1 Optimization ===")
    p1 = run_phase1_optimization(df)
    params1 = p1['params']
    fitness1 = p1['fitness']
    log.info(f"Phase1Params: {params1} (fitness={fitness1:.4f})")

    # Phase 2
    log.info("=== Phase 2 Optimization ===")
    p2 = run_phase2_optimization(df, base_params=params1)
    params2 = p2['params']
    fitness2 = p2['fitness']
    log.info(f"Phase2Params: {params2} (fitness={fitness2:.4f})")

    # Persist and backtest
    log_results(params1, params2, metrics=p2['metrics'], phase1_fitness=fitness1, phase2_fitness=fitness2)

if __name__ == "__main__":
    main()
