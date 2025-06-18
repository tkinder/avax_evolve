# src/core/results_logger.py

import sqlite3
from pathlib import Path
import pandas as pd
from src.core.backtester import backtest_strategy


def log_results(phase1, phase2, metrics, phase1_fitness, phase2_fitness, db_path="results.db"):
    """
    Persist Phase 1 & Phase 2 parameters and metrics to SQLite,
    then run a backtest on the stored OHLCV data for summary.
    """
    # Prepare results dictionary for insertion
    results = {
        "phase1_risk_reward": phase1.risk_reward,
        "phase1_trend": phase1.trend,
        "phase1_entry": phase1.entry,
        "phase1_confidence": phase1.confidence,
        "phase1_fitness": phase1_fitness,

        "phase2_risk_reward": phase2.risk_reward,
        "phase2_trend": phase2.trend,
        "phase2_entry": phase2.entry,
        "phase2_confidence": phase2.confidence,
        "phase2_bullish": phase2.bullish,
        "phase2_bearish": phase2.bearish,
        "phase2_top": phase2.top,
        "phase2_bottom": phase2.bottom,
        "phase2_neutral": phase2.neutral,
        "phase2_fitness": phase2_fitness,

        "returns": metrics.get("returns"),
        "sharpe": metrics.get("sharpe"),
        "volatility": metrics.get("volatility"),
        "max_drawdown": metrics.get("max_drawdown")
    }

    # Print summary of Phase results
    print("Final Results:")
    for k, v in results.items():
        try:
            print(f"{k}: {float(v):.4f}")
        except Exception:
            print(f"{k}: {v}")

    # Ensure database and table exist
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS strategy_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phase1_risk_reward REAL,
            phase1_trend REAL,
            phase1_entry REAL,
            phase1_confidence REAL,
            phase1_fitness REAL,

            phase2_risk_reward REAL,
            phase2_trend REAL,
            phase2_entry REAL,
            phase2_confidence REAL,
            phase2_bullish REAL,
            phase2_bearish REAL,
            phase2_top REAL,
            phase2_bottom REAL,
            phase2_neutral REAL,
            phase2_fitness REAL,

            returns REAL,
            sharpe REAL,
            volatility REAL,
            max_drawdown REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Insert results
    cursor.execute('''
        INSERT INTO strategy_runs (
            phase1_risk_reward, phase1_trend, phase1_entry, phase1_confidence, phase1_fitness,
            phase2_risk_reward, phase2_trend, phase2_entry, phase2_confidence,
            phase2_bullish, phase2_bearish, phase2_top, phase2_bottom, phase2_neutral, phase2_fitness,
            returns, sharpe, volatility, max_drawdown
        ) VALUES (
            :phase1_risk_reward, :phase1_trend, :phase1_entry, :phase1_confidence, :phase1_fitness,
            :phase2_risk_reward, :phase2_trend, :phase2_entry, :phase2_confidence,
            :phase2_bullish, :phase2_bearish, :phase2_top, :phase2_bottom, :phase2_neutral, :phase2_fitness,
            :returns, :sharpe, :volatility, :max_drawdown
        )
    ''', results)
    conn.commit()

    # Load OHLCV data for backtest summary
    try:
        conn2 = sqlite3.connect(db_path)
        df_ohlcv = pd.read_sql(
            "SELECT timestamp, open, high, low, close, volume FROM ohlcv_data ORDER BY timestamp", 
            conn2, parse_dates=["timestamp"]
        )
        conn2.close()
        if df_ohlcv.empty:
            print("⚠️ Backtest skipped: no data in 'ohlcv_data' table")
        else:
            print("Backtest Results:")
            bt_metrics = backtest_strategy(df_ohlcv, phase2)
            for k, v in bt_metrics.items():
                try:
                    print(f"{k}: {float(v):.4f}")
                except Exception:
                    print(f"{k}: {v}")
    except Exception as e:
        print(f"⚠️ Backtest skipped due to error: {e}")
    finally:
        conn.close()
