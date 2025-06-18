import sqlite3
from datetime import datetime
from dataclasses import asdict

def log_strategy_run(strategy_name, params, metrics, trades, wins, losses, final_balance, db_path="avax_backtest_results.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        strategy_name TEXT,
        returns REAL,
        sharpe REAL,
        volatility REAL,
        drawdown REAL,
        trades INTEGER,
        wins INTEGER,
        losses INTEGER,
        final_balance REAL,
        params_json TEXT
    )
    """)

    import json
    param_dict = asdict(params) if hasattr(params, '__dataclass_fields__') else params
    metrics = metrics or {}

    cursor.execute("""
    INSERT INTO runs (
        timestamp, strategy_name, returns, sharpe, volatility, drawdown,
        trades, wins, losses, final_balance, params_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        strategy_name,
        metrics.get("returns"),
        metrics.get("sharpe"),
        metrics.get("volatility"),
        metrics.get("max_drawdown"),
        trades,
        wins,
        losses,
        final_balance,
        json.dumps(param_dict)
    ))

    conn.commit()
    conn.close()
