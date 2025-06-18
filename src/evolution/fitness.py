# src/evolution/fitness.py

import numpy as np
import pandas as pd
from src.core.performance import get_performance_metrics
from src.core.params import Phase2Params
from src.evolution.utils import get_returns


def calculate_fitness(returns: pd.Series, risk_free_rate: float = 0.02 / 252) -> float:
    """
    Compute a composite fitness score that balances Sharpe ratio, drawdown, volatility,
    and absolute return by explicitly rewarding net gains.
    """
    if len(returns) == 0 or returns.isna().all():
        return 0.0

    # Clean and excess returns
    cleaned = returns.replace([np.inf, -np.inf], 0).fillna(0)
    excess_returns = cleaned - risk_free_rate

    # Sharpe calculation
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0

    # Cumulative returns and drawdown
    cumulative = (1 + cleaned).cumprod()
    drawdown = (cumulative.cummax() - cumulative) / cumulative.cummax()
    max_dd = drawdown.max() if len(drawdown) > 0 else 0.0

    # Volatility
    volatility = cleaned.std()

    # Total return (net PnL)
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0.0

    # Re-calibrated fitness: explicit reward for net returns
    score = (
        0.3 * np.tanh(sharpe / 3) +
        0.2 * (1 - max_dd) +
        0.2 * (1 - volatility) +
        0.3 * np.tanh(total_return * 10)
    )
    return max(0.0, round(score, 6))


def evaluate_strategy(df: pd.DataFrame, params: Phase2Params) -> dict:
    """
    Evaluate a Phase 2 strategy: backtest returns, compute fitness and performance metrics.
    """
    try:
        returns = get_returns(df)
        fitness = calculate_fitness(returns)
        metrics = get_performance_metrics(returns)
        metrics['fitness'] = fitness
        return metrics
    except Exception as e:
        print(f"[Strategy Eval Error] {e}")
        return {
            'fitness': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 1.0,
            'volatility': 1.0,
            'returns': 0.0
        }
