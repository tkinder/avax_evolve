# src/evolution/fitness.py

import numpy as np
import pandas as pd
from src.core.performance import get_performance_metrics
from src.core.params import Phase2Params
from src.evolution.utils import get_returns


def calculate_fitness(returns: pd.Series, risk_free_rate: float = 0.02 / 252) -> float:
    """
    Compute a composite fitness score based on Sharpe ratio, drawdown, and volatility.
    """
    if len(returns) == 0 or returns.isna().all():
        return 0.0

    cleaned = returns.replace([np.inf, -np.inf], 0).fillna(0)
    excess_returns = cleaned - risk_free_rate

    sharpe = (np.sqrt(252) * excess_returns.mean() / excess_returns.std()) \
        if excess_returns.std() > 0 else 0.0

    cumulative = (1 + cleaned).cumprod()
    drawdown = (cumulative.cummax() - cumulative) / cumulative.cummax()
    max_dd = drawdown.max() if len(drawdown) > 0 else 0.0

    volatility = cleaned.std()

    score = (
        0.4 * np.tanh(sharpe / 3) +
        0.3 * (1 - max_dd) +
        0.3 * (1 - volatility)
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
