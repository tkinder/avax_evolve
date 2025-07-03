# bitcoin_backtester.py
"""
Bitcoin-specific backtester using AVAX's proven validation approach
Adapted from AVAX's sophisticated backtest system
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from src.evolution.fitness import BacktestResult, calculate_backtest_fitness, FitnessConfig

@dataclass
class BitcoinParams:
    """
    Bitcoin-specific strategy parameters adapted from AVAX AdaptiveParams
    """
    # Core strategy parameters (from DEAP optimization)
    risk_reward: float = 1.5
    trend_strength: float = 1.0
    entry_threshold: float = 0.5
    confidence: float = 1.0
    
    # Bitcoin-specific thresholds (optimized by DEAP)
    buy_threshold_pct: float = 0.688    # Bitcoin's discovered optimal: 68.8%
    sell_threshold_pct: float = 0.697   # Bitcoin's discovered optimal: 69.7%
    
    # Regime detection parameters
    regime_short_window: int = 10
    regime_medium_window: int = 30
    regime_long_window: int = 60
    
    # Position sizing multipliers (from DEAP)
    bull_multiplier: float = 1.22
    bear_multiplier: float = 0.67
    high_vol_multiplier: float = 0.55
    low_vol_multiplier: float = 1.40
    
    # Risk management (Bitcoin-specific from DEAP)
    max_position_pct: float = 1.79      # Bitcoin allows higher leverage
    stop_loss_pct: float = 0.009        # Bitcoin's tight 0.9% stop
    take_profit_pct: float = 0.067      # Bitcoin's 6.7% target
    
    # Price level calculation
    price_lookback_days: int = 50

class BitcoinPriceLevels:
    """
    Calculate Bitcoin-specific adaptive buy/sell levels
    Adapted from AVAX's AdaptivePriceLevels
    """
    
    def __init__(self, params: BitcoinParams):
        self.params = params
    
    def get_price_levels(self, df: pd.DataFrame, current_idx: int, 
                        lookback_days: int = None) -> Tuple[float, float, float, float]:
        """
        Calculate Bitcoin-specific buy/sell levels based on recent price range.
        """
        if lookback_days is None:
            lookback_days = self.params.price_lookback_days
            
        if current_idx < lookback_days:
            lookback_days = current_idx
        
        if lookback_days < 10:
            # Fallback for insufficient data
            current_price = df.iloc[current_idx]['close']
            return current_price * 0.95, current_price * 1.05, current_price * 0.9, current_price * 1.1
        
        start_idx = max(0, current_idx - lookback_days)
        recent_data = df.iloc[start_idx:current_idx + 1]
        
        # Calculate price range
        price_low = recent_data['close'].min()
        price_high = recent_data['close'].max()
        price_range = price_high - price_low
        
        # Bitcoin-specific levels using DEAP-optimized thresholds
        buy_level = price_low + (price_range * self.params.buy_threshold_pct)
        sell_level = price_low + (price_range * self.params.sell_threshold_pct)
        
        # Bitcoin support/resistance (more conservative than AVAX)
        support_level = min(recent_data['close'].quantile(0.25), price_low * 1.01)  # Tighter for Bitcoin
        resistance_level = max(recent_data['close'].quantile(0.75), price_high * 0.99)
        
        # Ensure minimum spread
        if resistance_level - support_level < price_range * 0.05:  # Smaller spread for Bitcoin
            mid_price = (support_level + resistance_level) / 2
            support_level = mid_price - (price_range * 0.05)
            resistance_level = mid_price + (price_range * 0.05)
        
        return buy_level, sell_level, support_level, resistance_level

def bitcoin_backtest_strategy(df: pd.DataFrame, params: BitcoinParams) -> Dict[str, Any]:
    """
    Bitcoin-specific backtesting strategy using AVAX's proven approach
    """
    # Initialize Bitcoin price calculator
    price_calculator = BitcoinPriceLevels(params)
    
    # Trading variables (AVAX's proven setup)
    initial_balance = 10000
    balance = initial_balance
    position = 0  # 0 = no position, 1 = long
    position_size = 0
    entry_price = 0.0
    entry_index = 0
    
    # Performance tracking
    trades, wins, losses = 0, 0, 0
    peak_balance = initial_balance
    max_drawdown = 0
    equity_curve = []
    trade_log = []
    
    # Bitcoin-specific execution costs (more liquid than AVAX)
    fee_rate = 0.0005      # 0.05% fee (Bitcoin has lower fees)
    slippage_pct = 0.0002  # 0.02% slippage (Bitcoin more liquid)
    
    # Strategy loop (ensure enough data)
    start_idx = max(50, params.regime_long_window)
    
    # Add simple regime detection for Bitcoin
    df['sma_short'] = df['close'].rolling(window=params.regime_short_window).mean()
    df['sma_medium'] = df['close'].rolling(window=params.regime_medium_window).mean()
    df['sma_long'] = df['close'].rolling(window=params.regime_long_window).mean()
    
    for i in range(start_idx, len(df)):
        current_row = df.iloc[i]
        current_price = current_row['close']
        
        # Bitcoin regime detection (simplified but effective)
        sma_short = current_row.get('sma_short', current_price)
        sma_medium = current_row.get('sma_medium', current_price)
        sma_long = current_row.get('sma_long', current_price)
        
        # Determine Bitcoin market regime
        if pd.isna(sma_long):
            continue
            
        # Bitcoin trend classification
        if current_price > sma_long * 1.02 and sma_short > sma_medium:  # 2% buffer for Bitcoin
            bitcoin_trend = "bullish"
            regime_multiplier = params.bull_multiplier
        elif current_price < sma_long * 0.98 and sma_short < sma_medium:
            bitcoin_trend = "bearish"
            regime_multiplier = params.bear_multiplier
        else:
            bitcoin_trend = "neutral"
            regime_multiplier = 1.0
        
        # Bitcoin volatility assessment (last 14 days)
        if i >= 14:
            recent_returns = df.iloc[i-14:i]['close'].pct_change().dropna()
            volatility = recent_returns.std()
            if volatility > 0.04:  # High volatility for Bitcoin
                regime_multiplier *= params.high_vol_multiplier
            elif volatility < 0.015:  # Low volatility for Bitcoin
                regime_multiplier *= params.low_vol_multiplier
        
        # Calculate Bitcoin-specific price levels
        buy_level, sell_level, support, resistance = price_calculator.get_price_levels(df, i)
        
        # Fix support/resistance if needed
        if resistance <= support:
            support = current_price * 0.98
            resistance = current_price * 1.02
        
        # Position sizing
        base_position_size = balance * params.max_position_pct
        adjusted_position_size = base_position_size * regime_multiplier
        
        # ENTRY LOGIC (Bitcoin-specific)
        if position == 0:
            # Bitcoin entry conditions (very selective like discovered 68.8% threshold)
            price_range = max(resistance - support, current_price * 0.01)
            price_position = max(0, min(1, (current_price - support) / price_range))
            
            # Bitcoin only enters when deeply oversold (matching 68.8% buy threshold)
            bitcoin_oversold = price_position < (1 - params.buy_threshold_pct)  # Inverted because 68.8% means wait
            
            # Additional Bitcoin filters (conservative approach)
            trend_ok = bitcoin_trend in ["bullish", "neutral"]  # Avoid bearish entries
            
            # Bitcoin confidence filter (requires strong signals)
            confidence_ok = True
            if bitcoin_trend == "neutral":
                confidence_ok = price_position < 0.3  # Only if very oversold in neutral
            
            should_enter = bitcoin_oversold and trend_ok and confidence_ok
            
            if should_enter:
                # Execute Bitcoin buy
                executed_price = current_price * (1 + slippage_pct)
                position_size = adjusted_position_size / executed_price
                entry_price = executed_price
                entry_index = i
                position = 1
                balance -= adjusted_position_size * (1 + fee_rate)
                
                trades += 1
                trade_log.append({
                    'type': 'BUY',
                    'date': current_row.name,
                    'price': executed_price,
                    'size': position_size,
                    'trend': bitcoin_trend,
                    'regime_multiplier': regime_multiplier,
                    'price_position': price_position,
                    'buy_level': buy_level,
                    'sell_level': sell_level
                })
                
                print(f"🟢 BTC BUY at ${executed_price:,.2f} on {current_row.name} "
                      f"(trend: {bitcoin_trend}, pos: {price_position:.2f})")
        
        else:  # In position
            # EXIT LOGIC (Bitcoin-specific - very tight like 0.9% stop)
            current_value = position_size * current_price
            unrealized_pnl = (current_price - entry_price) / entry_price
            days_in_position = i - entry_index
            
            should_exit = False
            exit_reason = ""
            
            # Bitcoin stop loss (tight 0.9% like DEAP discovered)
            if unrealized_pnl <= -params.stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Bitcoin take profit (6.7% like DEAP discovered)
            elif unrealized_pnl >= params.take_profit_pct:
                should_exit = True
                exit_reason = "take_profit"
            
            # Bitcoin sell threshold (69.7% like DEAP discovered)
            else:
                price_range = max(resistance - support, current_price * 0.01)
                price_position = max(0, min(1, (current_price - support) / price_range))
                
                if price_position > params.sell_threshold_pct:  # 69.7% threshold
                    should_exit = True
                    exit_reason = "sell_threshold"
            
            # Bitcoin emergency exits
            if bitcoin_trend == "bearish" and unrealized_pnl < -0.03:  # 3% loss in bear
                should_exit = True
                exit_reason = "bear_emergency"
            
            # Bitcoin trailing stop for big gains
            if unrealized_pnl > 0.15:  # 15%+ profit
                recent_high_idx = max(0, i-3)  # Last 3 days for Bitcoin
                recent_high = df.iloc[recent_high_idx:i+1]['close'].max()
                if current_price < recent_high * 0.98:  # 2% trailing for Bitcoin
                    should_exit = True
                    exit_reason = "trailing_stop"
            
            if should_exit:
                # Execute Bitcoin sell
                executed_price = current_price * (1 - slippage_pct)
                proceeds = position_size * executed_price * (1 - fee_rate)
                balance += proceeds
                
                # Record trade result
                gross_return = (executed_price - entry_price) / entry_price
                is_win = gross_return > 0
                wins += is_win
                losses += not is_win
                
                trade_log.append({
                    'type': 'SELL',
                    'date': current_row.name,
                    'price': executed_price,
                    'size': position_size,
                    'return': gross_return,
                    'reason': exit_reason,
                    'trend': bitcoin_trend,
                    'days_held': days_in_position,
                    'is_win': is_win
                })
                
                print(f"🔴 BTC SELL at ${executed_price:,.2f} on {current_row.name} "
                      f"(reason: {exit_reason}, return: {gross_return:.1%}, days: {days_in_position})")
                
                # Reset position
                position = 0
                position_size = 0
                entry_price = 0.0
                entry_index = 0
        
        # Update equity curve
        if position == 1:
            current_value = position_size * current_price
            total_equity = balance + current_value
        else:
            total_equity = balance
        
        equity_curve.append(total_equity)
        
        # Update drawdown
        peak_balance = max(peak_balance, total_equity)
        current_drawdown = (peak_balance - total_equity) / peak_balance
        max_drawdown = max(max_drawdown, current_drawdown)
    
    # Force exit if still in position
    if position == 1:
        final_price = df.iloc[-1]['close'] * (1 - slippage_pct)
        proceeds = position_size * final_price * (1 - fee_rate)
        balance += proceeds
        
        gross_return = (final_price - entry_price) / entry_price
        is_win = gross_return > 0
        wins += is_win
        losses += not is_win
        
        trade_log.append({
            'type': 'FORCED_SELL',
            'date': df.index[-1],
            'price': final_price,
            'size': position_size,
            'return': gross_return,
            'reason': 'end_of_data',
            'days_held': len(df) - 1 - entry_index,
            'is_win': is_win
        })
    
    # Calculate final metrics
    final_return = (balance - initial_balance) / initial_balance
    
    # Calculate performance metrics
    if len(equity_curve) > 1:
        equity_series = pd.Series(equity_curve)
        returns_series = equity_series.pct_change().dropna()
        
        if len(returns_series) > 0 and returns_series.std() > 0:
            sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
            volatility = returns_series.std() * np.sqrt(252)
        else:
            sharpe = 0
            volatility = 0
    else:
        sharpe = 0
        volatility = 0
    
    # Calculate win rate
    win_rate = wins / trades if trades > 0 else 0
    
    print(f"\n₿ Bitcoin Strategy Results:")
    print(f"   Trades: {trades}, Wins: {wins}, Losses: {losses}")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Final Balance: ${balance:,.2f}")
    print(f"   Total Return: {final_return:.1%}")
    print(f"   Max Drawdown: {max_drawdown:.1%}")
    print(f"   Sharpe Ratio: {sharpe:.4f}")
    
    return {
        'returns': final_return,
        'sharpe': sharpe,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'final_balance': balance,
        'trade_log': trade_log,
        'equity_curve': equity_curve
    }

def evaluate_bitcoin_strategy_with_backtest(df: pd.DataFrame, params: BitcoinParams, 
                                          fitness_config=None) -> Dict[str, Any]:
    """
    Evaluate Bitcoin strategy using AVAX's proven backtest validation approach
    """
    try:
        # Run Bitcoin backtest
        results = bitcoin_backtest_strategy(df, params)
        
        # Create BacktestResult object (AVAX's approach)
        equity_series = pd.Series(results['equity_curve'])
        daily_returns = equity_series.pct_change().fillna(0)
        
        backtest_result = BacktestResult(
            final_balance=results['final_balance'],
            initial_balance=10000,
            trade_count=results['trades'],
            winning_trades=results['wins'],
            losing_trades=results['losses'],
            max_drawdown=results['max_drawdown'],
            returns=daily_returns
        )
        
        # Calculate fitness using AVAX's proven method
        if fitness_config is None:
            fitness_config = FitnessConfig(
                # Bitcoin-specific fitness config (conservative)
                min_trades=3,                    # Bitcoin trades less frequently
                max_drawdown_threshold=0.35,     # Strict drawdown for Bitcoin
                min_profit_threshold=-0.10,      # Limited losses
                profitability_weight=0.35,       # Focus on profits
                risk_adjusted_weight=0.35,       # Risk-adjusted returns important
                drawdown_weight=0.20,            # Drawdown control
                trade_quality_weight=0.10        # Trade quality
            )
        
        fitness, components = calculate_backtest_fitness(backtest_result, fitness_config)
        
        # Combine results (AVAX style)
        results['fitness'] = fitness
        results.update(components)
        
        return results
        
    except Exception as e:
        print(f"[Bitcoin Strategy Error] {e}")
        return {
            'fitness': 0.01,
            'returns': -1.0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'final_balance': 5000,
            'max_drawdown': 1.0,
            'error': str(e)
        }

# Test function
def test_bitcoin_backtester():
    """
    Test the enhanced Bitcoin backtester
    """
    from bitcoin_data_pipeline import BitcoinDataPipeline
    
    print("🧪 Testing Enhanced Bitcoin Backtester...")
    
    # Get Bitcoin data
    pipeline = BitcoinDataPipeline()
    df = pipeline.get_complete_dataset(historical_days=90)
    
    if df is None:
        print("❌ No Bitcoin data available")
        return
    
    # Test with DEAP-optimized parameters
    test_params = BitcoinParams(
        buy_threshold_pct=0.688,    # From DEAP optimization
        sell_threshold_pct=0.697,   # From DEAP optimization
        stop_loss_pct=0.009,        # From DEAP optimization
        take_profit_pct=0.067,      # From DEAP optimization
        max_position_pct=1.79,      # From DEAP optimization
        bull_multiplier=1.22,       # From DEAP optimization
        bear_multiplier=0.67,       # From DEAP optimization
        high_vol_multiplier=0.55,   # From DEAP optimization
        low_vol_multiplier=1.40     # From DEAP optimization
    )
    
    # Run enhanced backtest
    results = evaluate_bitcoin_strategy_with_backtest(df, test_params)
    
    print(f"\n✅ Enhanced Bitcoin Backtester Results:")
    print(f"   Fitness Score: {results['fitness']:.4f}")
    print(f"   Return: {results['returns']:.1%}")
    print(f"   Trades: {results['trades']}")
    print(f"   Win Rate: {results.get('win_rate', 0):.1%}")
    print(f"   Max Drawdown: {results['max_drawdown']:.1%}")

if __name__ == "__main__":
    test_bitcoin_backtester()
