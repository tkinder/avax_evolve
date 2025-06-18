# core/performance.py
import numpy as np
import pandas as pd

def get_performance_metrics(returns: pd.Series) -> dict:
    returns = returns.replace([np.inf, -np.inf], 0).fillna(0)

    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

    cumulative = (1 + returns).cumprod()
    drawdown = (cumulative.cummax() - cumulative) / cumulative.cummax()
    max_dd = drawdown.max() if len(drawdown) > 0 else 0

    volatility = returns.std()
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0

    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "volatility": float(volatility),
        "returns": float(total_return),
    }
