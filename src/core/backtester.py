# src/core/backtester.py

import pandas as pd
from src.core.strategy_logger import log_strategy_run

# ─── Cost parameters ───────────────────────────────────────────────────────────
COMMISSION_RATE = 0.001   # 0.1% per side
SLIPPAGE_PCT    = 0.0005  # 0.05% slippage assuming mid-price fill
# ────────────────────────────────────────────────────────────────────────────────

def backtest_strategy(df: pd.DataFrame, params):
    """
    Execute backtest with simple regime filter and trailing stop,
    accounting for commissions and slippage.
    """
    df = df.copy()
    df['sma200'] = df['close'].rolling(window=200).mean()
    df['regime'] = df['close'] > df['sma200']

    initial_balance = 10000.0
    balance = initial_balance
    position = 0
    entry_price = 0.0
    highest_price = 0.0

    trades = 0
    wins = 0
    losses = 0

    equity_curve = []
    peak_balance = initial_balance
    max_drawdown = 0.0

    trailing_stop_pct = 0.10

    for idx, row in df.iterrows():
        price = row['close']

        # Not enough data yet
        if pd.isna(row['sma200']):
            equity_curve.append(balance)
            continue

        in_uptrend = row['regime']

        if position == 0:
            # BUY signal
            if price < params.bottom and in_uptrend:
                # apply slippage on entry
                entry_price = price * (1 + SLIPPAGE_PCT)
                # commission on entry
                balance -= entry_price * COMMISSION_RATE
                position = 1
                highest_price = entry_price
                trades += 1
        else:
            # update highest
            highest_price = max(highest_price, price)
            stop_price = highest_price * (1 - trailing_stop_pct)

            # SELL conditions
            if price > params.top or price < stop_price:
                # apply slippage on exit
                exit_price = price * (1 - SLIPPAGE_PCT)
                # compute gross return
                gross_return = (exit_price - entry_price) / entry_price
                # subtract entry+exit commissions
                net_return = gross_return - 2 * COMMISSION_RATE
                balance *= (1 + net_return)
                # log win/loss
                wins += 1 if net_return > 0 else 0
                losses += 1 if net_return < 0 else 0
                position = 0

        # track equity
        equity_curve.append(balance)
        peak_balance = max(peak_balance, balance)
        drawdown = (peak_balance - balance) / peak_balance
        max_drawdown = max(max_drawdown, drawdown)

    # handle open position at end
    if position == 1:
        final_price = df.iloc[-1]['close']
        exit_price = final_price * (1 - SLIPPAGE_PCT)
        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - 2 * COMMISSION_RATE
        balance *= (1 + net_return)
        wins += 1 if net_return > 0 else 0
        losses += 1 if net_return < 0 else 0

    # return metrics
    returns = (pd.Series(equity_curve).pct_change().fillna(0))
    volatility = returns.std() * (252**0.5)
    sharpe = returns.mean() / returns.std() * (252**0.5) if returns.std() > 0 else 0.0

    metrics = {
        'returns': (balance - initial_balance) / initial_balance,
        'final_balance': balance,
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'sharpe': sharpe,
        'volatility': volatility,
        'max_drawdown': max_drawdown
    }

    # log to database
    log_strategy_run(
        strategy_name="avax_regime_v1",
        params=params,
        metrics=metrics,
        trades=trades,
        wins=wins,
        losses=losses,
        final_balance=balance
    )

    return metrics
