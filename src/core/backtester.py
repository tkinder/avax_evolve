# src/core/backtester.py
import pandas as pd
import numpy as np
from src.core.strategy_logger import log_strategy_run
from src.evolution.fitness import BacktestResult, calculate_backtest_fitness, FitnessConfig

def backtest_strategy(df, params, return_detailed=False):
    """
    Enhanced backtest that can return either legacy metrics or detailed BacktestResult.
    
    Args:
        df: Price data DataFrame
        params: Strategy parameters
        return_detailed: If True, returns BacktestResult object for new fitness calculation
        
    Returns:
        Either legacy metrics dict or BacktestResult object
    """
    # realistic execution costs
    fee_rate     = 0.001   # 0.1% fee per side
    slippage_pct = 0.0005  # 0.05% slippage per trade

    # normalize top/bottom if given as fractions
    if params.top < 5 and params.bottom < 5:
        price_min = df['close'].min()
        price_max = df['close'].max()
        span      = price_max - price_min
        params.bottom = price_min + span * params.bottom
        params.top    = price_min + span * params.top
    if params.top < params.bottom:
        params.top, params.bottom = params.bottom, params.top

    print(f"Params: top={params.top:.4f}, bottom={params.bottom:.4f}")
    print(f"Price range: min={df['close'].min():.4f}, max={df['close'].max():.4f}")

    df['sma200']  = df['close'].rolling(window=200).mean()
    df['regime']  = df['close'] > df['sma200']

    initial_balance = 10000
    balance         = initial_balance
    position        = 0
    entry_price     = 0.0
    trades, wins, losses = 0, 0, 0
    peak_balance    = initial_balance
    max_drawdown    = 0
    equity_curve    = []
    
    # Track individual trades for detailed analysis
    trade_log = []
    daily_returns = []

    trailing_stop_pct = 0.1
    highest_price     = 0

    for i in range(1, len(df)):
        row   = df.iloc[i]
        close = row['close']
        if pd.isna(row['sma200']):
            equity_curve.append(balance)
            daily_returns.append(0.0)  # No change if no SMA
            continue

        in_uptrend = row['regime']
        prev_balance = balance

        if position == 0:
            if close < params.bottom and in_uptrend:
                executed_price = close * (1 + slippage_pct)
                print(f"BUY at {executed_price:.2f} on {row.name} (incl. slippage)")
                position    = 1
                entry_price = executed_price
                highest_price = close
                trades += 1
                
                # Log trade entry
                trade_log.append({
                    'type': 'BUY',
                    'price': executed_price,
                    'date': row.name,
                    'index': i
                })
                
        else:
            highest_price = max(highest_price, close)
            stop_price    = highest_price * (1 - trailing_stop_pct)

            if close > params.top or close < stop_price:
                executed_price = close * (1 - slippage_pct)
                print(f"SELL at {executed_price:.2f} on {row.name} (incl. slippage)")

                gross_ret = (executed_price - entry_price) / entry_price
                net_ret   = gross_ret - 2 * fee_rate
                balance  *= (1 + net_ret)

                is_win = gross_ret > 0
                wins   += 1 if is_win else 0
                losses += 1 if not is_win else 0
                position = 0
                
                # Log trade exit
                trade_log.append({
                    'type': 'SELL',
                    'price': executed_price,
                    'date': row.name,
                    'index': i,
                    'gross_return': gross_ret,
                    'net_return': net_ret,
                    'is_win': is_win
                })

        # Calculate daily return
        daily_return = (balance - prev_balance) / prev_balance if prev_balance > 0 else 0.0
        daily_returns.append(daily_return)
        
        equity_curve.append(balance)
        peak_balance = max(peak_balance, balance)
        drawdown     = (peak_balance - balance) / peak_balance
        max_drawdown = max(max_drawdown, drawdown)

    # forced exit if still in a position
    if position == 1:
        last_close     = df.iloc[-1]['close']
        executed_price = last_close * (1 - slippage_pct)
        print(f"FORCED SELL at {executed_price:.2f} on {df.index[-1]} (incl. slippage)")

        gross_ret = (executed_price - entry_price) / entry_price
        net_ret   = gross_ret - 2 * fee_rate
        balance  *= (1 + net_ret)

        is_win = gross_ret > 0
        wins   += 1 if is_win else 0
        losses += 1 if not is_win else 0
        
        # Log forced exit
        trade_log.append({
            'type': 'FORCED_SELL',
            'price': executed_price,
            'date': df.index[-1],
            'index': len(df) - 1,
            'gross_return': gross_ret,
            'net_return': net_ret,
            'is_win': is_win
        })

    print(f"Trades: {trades}, Wins: {wins}, Losses: {losses}, Final Balance: {balance:.2f}")

    # Calculate legacy metrics
    returns   = (balance - initial_balance) / initial_balance
    pct_change = pd.Series(equity_curve).pct_change().fillna(0)
    volatility = pct_change.std() * (252**0.5)
    sharpe     = (pct_change.mean() / volatility) * (252**0.5) if volatility else 0

    legacy_metrics = {
        "returns": returns,
        "sharpe": sharpe,
        "volatility": volatility,
        "max_drawdown": max_drawdown
    }

    # Log strategy run (existing functionality)
    log_strategy_run(
        strategy_name="avax_regime_v1",
        params=params,
        metrics=legacy_metrics,
        trades=trades,
        wins=wins,
        losses=losses,
        final_balance=balance
    )

    # Return appropriate format based on request
    if return_detailed:
        # Create returns series for new fitness calculation
        returns_series = pd.Series(daily_returns, index=df.index[:len(daily_returns)])
        
        # Return BacktestResult for new fitness function
        return BacktestResult(
            final_balance=balance,
            initial_balance=initial_balance,
            trade_count=trades,
            winning_trades=wins,
            losing_trades=losses,
            max_drawdown=max_drawdown,
            returns=returns_series,
            trades_executed=trade_log
        )
    else:
        # Return legacy metrics for backward compatibility
        return legacy_metrics


def evaluate_strategy_with_new_fitness(df, params, config=None):
    """
    Evaluate strategy using the new backtest-driven fitness function.
    
    This replaces your old evaluate_strategy function for the evolutionary algorithm.
    
    Args:
        df: Price data
        params: Strategy parameters
        config: FitnessConfig object (optional)
        
    Returns:
        Dictionary with fitness and detailed metrics
    """
    try:
        # Run backtest and get detailed results
        backtest_result = backtest_strategy(df, params, return_detailed=True)
        
        # Calculate fitness using actual backtest performance
        fitness, components = calculate_backtest_fitness(backtest_result, config)
        
        # Combine all metrics
        metrics = {
            'fitness': fitness,
            'total_return': backtest_result.total_return,
            'final_balance': backtest_result.final_balance,
            'trade_count': backtest_result.trade_count,
            'win_rate': backtest_result.win_rate,
            'max_drawdown': backtest_result.max_drawdown,
            'profit_factor': backtest_result.profit_factor,
            
            # Component breakdown for analysis
            'profitability_component': components.get('profitability', 0),
            'risk_adjusted_component': components.get('risk_adjusted', 0),
            'drawdown_component': components.get('drawdown', 0),
            'trade_quality_component': components.get('trade_quality', 0),
            
            # Legacy metrics for compatibility
            'returns': backtest_result.total_return,
            'sharpe': 0.0,  # Will be calculated from returns if available
            'volatility': 0.0  # Will be calculated from returns if available
        }
        
        # Calculate traditional metrics if we have return series
        if backtest_result.returns is not None and len(backtest_result.returns) > 0:
            try:
                pct_change = backtest_result.returns.fillna(0)
                volatility = pct_change.std() * (252**0.5) if pct_change.std() > 0 else 0
                sharpe = (pct_change.mean() / volatility) * (252**0.5) if volatility > 0 else 0
                
                metrics['sharpe'] = sharpe
                metrics['volatility'] = volatility
            except Exception as e:
                print(f"[Traditional Metrics Warning] {e}")
        
        return metrics
        
    except Exception as e:
        print(f"[Strategy Evaluation Error] {e}")
        config = config or FitnessConfig()
        return {
            'fitness': config.min_fitness,
            'total_return': -1.0,
            'final_balance': 0.0,
            'trade_count': 0,
            'win_rate': 0.0,
            'max_drawdown': 1.0,
            'profit_factor': 0.0,
            'error': str(e)
        }


# Wrapper function that matches your existing interface
def your_backtest_func(df, params):
    """
    Simple wrapper that returns BacktestResult for the new fitness system.
    This is what you'd pass to optimize_with_deap or other optimization functions.
    """
    return backtest_strategy(df, params, return_detailed=True)


# ============================================================================
# Example Usage and Migration Guide
# ============================================================================

def example_migration():
    """
    Example showing how to migrate from old to new fitness calculation.
    """
    # OLD WAY (what you were doing before):
    # def evaluate_strategy(df, params):
    #     returns = get_returns(df)  # Synthetic returns
    #     fitness = calculate_fitness(returns)  # Didn't match backtest
    #     metrics = get_performance_metrics(returns)
    #     metrics['fitness'] = fitness
    #     return metrics
    
    # NEW WAY (what you should do now):
    def evaluate_strategy_new(df, params):
        return evaluate_strategy_with_new_fitness(df, params)
    
    # For DEAP optimization:
    def fitness_func_for_deap(param_dict):
        # Convert param_dict to your Phase2Params object
        # params = Phase2Params(**param_dict)
        
        # Use the new evaluation
        # metrics = evaluate_strategy_new(df, params)
        # return metrics['fitness']
        pass
    
    print("Migration complete! Replace your evaluate_strategy calls with evaluate_strategy_with_new_fitness")


if __name__ == "__main__":
    print("🔧 Enhanced Backtester Ready!")
    print("   - Use backtest_strategy(df, params, return_detailed=False) for legacy mode")
    print("   - Use backtest_strategy(df, params, return_detailed=True) for new fitness mode")
    print("   - Use evaluate_strategy_with_new_fitness(df, params) to replace old evaluate_strategy")
    print("   - Use your_backtest_func(df, params) for DEAP optimization")