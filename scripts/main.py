#!/usr/bin/env python3
# scripts/main.py

import subprocess
from src.core.data import fetch_historical_data
from src.core.logging import get_logger
from src.core.results_logger import log_results
from src.evolution.phase1 import run_phase1_optimization
from src.evolution.phase2 import run_phase2_optimization


def main():
    # 1) Refresh OHLCV data
    subprocess.run(["python", "scripts/fetch_and_store_ohlcv.py"], check=True)
    
    # 2) Load data
    df = fetch_historical_data(refresh=False)
    
    # 3) Phase 1 optimization
    log = get_logger("avax_evolve")
    log.info("=== Phase 1 Optimization ===")
    p1 = run_phase1_optimization(df)
    params1, fitness1 = p1['params'], p1['fitness']
    log.info(f"Phase 1 best params: {params1} (fitness={fitness1:.4f})")
    
    # 4) Phase 2 optimization
    log.info("=== Phase 2 Optimization ===")
    p2 = run_phase2_optimization(df, base_params=params1)
    params2, metrics2, fitness2 = p2['params'], p2['metrics'], p2['fitness']
    log.info(f"Phase 2 best params: {params2} (fitness={fitness2:.4f})")
    
    # 5) Persist results
    log_results(params1, params2, metrics2, fitness1, fitness2)


if __name__ == "__main__":
    main()
