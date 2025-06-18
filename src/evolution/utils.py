# evolution/utils.py
import pandas as pd
import numpy as np

def get_returns(historical_data: pd.DataFrame) -> pd.Series:
    return historical_data['close'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

def get_performance_metrics(returns: pd.Series) -> dict:
    returns = returns.replace([np.inf, -np.inf], 0).fillna(0)
    cumulative = (1 + returns).cumprod()
    drawdown = (cumulative.cummax() - cumulative) / cumulative.cummax()

    return {
        'sharpe': float(np.sqrt(252) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0,
        'max_drawdown': float(drawdown.max() if len(drawdown) > 0 else 0.0),
        'volatility': float(returns.std()),
        'returns': float(returns.sum())
    }
