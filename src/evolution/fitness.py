# fitness.py
from src.core.performance import get_performance_metrics
from src.core.params import Phase2Params
from core.backtester import backtest_strategy
import numpy as np

def calculate_fitness(returns):
    """
    Composite fitness score combining Sharpe, drawdown, and volatility.
    """
    try:
        metrics = get_performance_metrics(returns)
        sharpe = metrics['sharpe']
        max_dd = metrics['max_drawdown']
        volatility = metrics['volatility']

        score = (
            0.4 * np.tanh(sharpe / 3) +
            0.3 * (1 - max_dd) +
            0.3 * (1 - volatility)
        )

        return round(score, 4)
    except Exception as e:
        print(f"[Fitness Error] {e}")
        return 0.0

def evaluate_strategy(df, params: Phase2Params) -> dict:
    """
    Run the actual backtest strategy and return evaluation metrics.
    """
    try:
        result = backtest_strategy(df.copy(), params)
        returns = result["returns"]
        fitness = calculate_fitness(returns)
        metrics = get_performance_metrics(returns)
        metrics.update({
            "fitness": fitness,
            "final_balance": result["final_balance"],
            "num_trades": result["num_trades"],
            "wins": result["wins"],
            "losses": result["losses"]
        })
        return metrics
    except Exception as e:
        print(f"[Strategy Eval Error] {e}")
        return {
            "fitness": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 1.0,
            "volatility": 1.0,
            "returns": 0.0,
            "final_balance": 10000.0,
            "num_trades": 0,
            "wins": 0,
            "losses": 0
        }
