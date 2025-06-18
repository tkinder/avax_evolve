#!/usr/bin/env python3
# scripts/view_logged_strategies.py

import sqlite3
import pandas as pd
from tabulate import tabulate
from pathlib import Path
import argparse

# Default path to the SQLite DB (project root/results.db)
DEFAULT_DB = Path(__file__).parent.parent / "results.db"


def view_logged_strategies(db_path=None, sort_by='timestamp', desc=True, limit=20):
    """
    Display recent strategy runs sorted by a given column.
    """
    db = db_path or DEFAULT_DB
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = '''
        SELECT
            id, timestamp,
            phase1_risk_reward, phase1_trend, phase1_entry, phase1_confidence, phase1_fitness,
            phase2_risk_reward, phase2_trend, phase2_entry, phase2_confidence,
            phase2_bullish, phase2_bearish, phase2_top, phase2_bottom, phase2_neutral,
            phase2_fitness, returns, sharpe, volatility, max_drawdown
        FROM strategy_runs
    '''

    valid_columns = [
        'id', 'timestamp', 'phase1_risk_reward', 'phase1_trend', 'phase1_entry',
        'phase1_confidence', 'phase1_fitness', 'phase2_risk_reward', 'phase2_trend',
        'phase2_entry', 'phase2_confidence', 'phase2_bullish', 'phase2_bearish',
        'phase2_top', 'phase2_bottom', 'phase2_neutral', 'phase2_fitness',
        'returns', 'sharpe', 'volatility', 'max_drawdown'
    ]

    if sort_by not in valid_columns:
        print(f"Warning: '{sort_by}' is not valid; defaulting to 'timestamp'.")
        sort_by = 'timestamp'

    query += f" ORDER BY {sort_by} {'DESC' if desc else 'ASC'} LIMIT {limit}"  
    cursor.execute(query)
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]

    df = pd.DataFrame(rows, columns=cols)
    print(f"\nShowing {len(df)} runs sorted by '{sort_by}' ({'DESC' if desc else 'ASC'})")
    print("=" * 80)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, floatfmt='.4f'))

    if not df.empty:
        print("\nSummary Statistics:")
        print(f"  Average Returns: {df['returns'].mean():.4f}")
        best = df.loc[df['returns'].idxmax()]
        print(f"  Best Returns: {best['returns']:.4f} (ID: {int(best['id'])})")
        worst = df.loc[df['returns'].idxmin()]
        print(f"  Worst Returns: {worst['returns']:.4f} (ID: {int(worst['id'])})")
        print(f"  Average Sharpe: {df['sharpe'].mean():.4f}")
        top_sharpe = df.loc[df['sharpe'].idxmax()]
        print(f"  Best Sharpe: {top_sharpe['sharpe']:.4f} (ID: {int(top_sharpe['id'])})")
        print(f"  Average Max Drawdown: {df['max_drawdown'].mean():.4f}")

    conn.close()


def view_top_performers(db_path=None, metric='returns', limit=10):
    """
    Display top-performing strategies by a specified metric.
    """
    db = db_path or DEFAULT_DB
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = f'''
        SELECT id, timestamp, {metric}, returns, sharpe, volatility, max_drawdown,
               phase1_fitness, phase2_fitness
        FROM strategy_runs
        ORDER BY {metric} DESC LIMIT {limit}
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]

    df = pd.DataFrame(rows, columns=cols)
    print(f"\nTop {limit} strategies by '{metric}':")
    print("=" * 60)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, floatfmt='.4f'))
    conn.close()


def view_recent_runs(db_path=None, hours=24, limit=20):
    """
    Display strategy runs from the last N hours.
    """
    db = db_path or DEFAULT_DB
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = f'''
        SELECT id, timestamp, returns, sharpe, volatility, max_drawdown,
               phase1_fitness, phase2_fitness
        FROM strategy_runs
        WHERE timestamp >= datetime('now', '-{hours} hours')
        ORDER BY timestamp DESC LIMIT {limit}
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]

    df = pd.DataFrame(rows, columns=cols)
    print(f"\nStrategy runs in last {hours} hours (limit {limit}):")
    print("=" * 60)
    if df.empty:
        print("No runs found in the specified period.")
    else:
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, floatfmt='.4f'))

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View AVAX Evolve strategy runs")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subcommand: list
    p1 = subparsers.add_parser('list', help='List recent runs')
    p1.add_argument('--limit', type=int, default=20)
    p1.add_argument('--sort_by', default='timestamp')
    p1.add_argument('--desc', action='store_true', help='Sort descending')
    p1.add_argument('--asc', action='store_true', help='Sort ascending')

    # Subcommand: top
    p2 = subparsers.add_parser('top', help='Show top performers')
    p2.add_argument('--metric', default='returns')
    p2.add_argument('--limit', type=int, default=10)

    # Subcommand: recent
    p3 = subparsers.add_parser('recent', help='Show runs from last N hours')
    p3.add_argument('--hours', type=int, default=24)
    p3.add_argument('--limit', type=int, default=20)

    args = parser.parse_args()
    if args.command == 'list':
        sort_desc = args.desc or not args.asc
        view_logged_strategies(sort_by=args.sort_by, desc=sort_desc, limit=args.limit)
    elif args.command == 'top':
        view_top_performers(metric=args.metric, limit=args.limit)
    elif args.command == 'recent':
        view_recent_runs(hours=args.hours, limit=args.limit)
