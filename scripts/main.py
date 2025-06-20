#!/usr/bin/env python3
# scripts/main.py - Speed Optimized Version

import sys
from pathlib import Path
# Ensure project root and src folder are on PYTHONPATH so modules can be imported
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import subprocess
import sqlite3
from src.core.data import fetch_historical_data
from src.core.logging import get_logger
from src.core.results_logger import log_results

# Import the new, working optimization functions
from src.evolution.phase1_fixed import run_phase1_optimization
from src.evolution.phase2_fixed import run_phase2_optimization

# Bootstrap OHLCV data
DB_PATH = project_root / "results.db"
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

# Single pipeline execution - SPEED OPTIMIZED
def main():
    log.info("🔥 Starting pipeline run")
    
    # Load data
    df = fetch_historical_data(refresh=False)
    log.info(f"📊 Loaded {len(df)} rows of historical data")

    # Phase 1 - REDUCED PARAMETERS for speed
    log.info("=== Phase 1 Optimization ===")
    p1 = run_phase1_optimization(
        df, 
        ngen=10,      # Reduced from 20
        pop_size=30,  # Reduced from 50
        log=log
    )
    params1 = p1['params']
    fitness1 = p1['fitness']
    log.info(f"Phase1Params: {params1} (fitness={fitness1:.4f})")

    # Phase 2 - REDUCED PARAMETERS for speed
    log.info("=== Phase 2 Optimization ===")
    p2 = run_phase2_optimization(
        df, 
        base_params=params1, 
        ngen=15,      # Reduced from 25
        pop_size=40,  # Reduced from 60
        log=log
    )
    params2 = p2['params']
    fitness2 = p2['fitness']
    log.info(f"Phase2Params: {params2} (fitness={fitness2:.4f})")

    # Persist and backtest - this should now MATCH the optimization results
    log_results(params1, params2, metrics=p2['metrics'], phase1_fitness=fitness1, phase2_fitness=fitness2)

if __name__ == "__main__":
    main()