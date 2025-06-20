# src/core/adaptive_backtester.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from src.core.strategy_logger import log_strategy_run
from src.core.regime_detector import (
    EnhancedRegimeDetector, 
    MarketRegime, 
    RegimeBasedTradingRules
)

@dataclass
class AdaptiveParams:
    """
    Adaptive strategy parameters using relative values instead of absolute prices.
    """
    # Core strategy parameters
    risk_reward: float = 1.5
    trend_strength: float = 1.0
    entry_threshold: float = 0.5
    confidence: float = 1.0
    
    # Relative price thresholds (as percentiles of recent price range)
    buy_threshold_pct: float = 0.3    # Buy when price is in bottom 30% of recent range
    sell_threshold_pct: float = 0.7   # Sell when price is in top 70% of recent range
    
    # Enhanced regime detection parameters
    regime_short_window: int = 10     # Short window for regime detection
    regime_medium_window: int = 30    # Medium window for regime detection  
    regime_long_window: int = 60      # Long window for regime detection
    
    # Legacy parameters (kept for backward compatibility with optimization scripts)
    trend_lookback: int = 20          # Days to determine trend
    volatility_lookback: int = 14     # Days to measure volatility
    volume_lookback: int = 10         # Days to assess volume
    bull_multiplier: float = 1.2      # Increase position in bull markets (legacy)
    bear_multiplier: float = 0.8      # Reduce position in bear markets (legacy)
    high_vol_multiplier: float = 0.7  # Reduce position in high volatility (legacy)
    low_vol_multiplier: float = 1.1   # Increase position in low volatility (legacy)
    
    # Risk management
    max_position_pct: float = 0.95    # Max % of balance to risk
    stop_loss_pct: float = 0.08       # 8% stop loss
    take_profit_pct: float = 0.15     # 15% take profit
    
    # Price level calculation
    price_lookback_days: int = 50     # Days for price level calculation

class AdaptivePriceLevels:
    """
    Calculate adaptive buy/sell levels based on recent price action.
    """
    
    def __init__(self, params: AdaptiveParams):
        self.params = params
    
    def get_price_levels(self, df: pd.DataFrame, current_idx: int, 
                        lookback_days: int = None) -> Tuple[float, float, float, float]:
        """
        Calculate adaptive buy/sell levels based on recent price range.
        
        Returns:
            buy_level, sell_level, support_level, resistance_level
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
        
        # Adaptive levels based on percentiles
        buy_level = price_low + (price_range * self.params.buy_threshold_pct)
        sell_level = price_low + (price_range * self.params.sell_threshold_pct)
        
        # Support/resistance levels using more robust calculation
        support_level = min(recent_data['close'].quantile(0.2), price_low * 1.02)
        resistance_level = max(recent_data['close'].quantile(0.8), price_high * 0.98)
        
        # Ensure minimum spread between support and resistance
        if resistance_level - support_level < price_range * 0.1:
            mid_price = (support_level + resistance_level) / 2
            support_level = mid_price - (price_range * 0.1)
            resistance_level = mid_price + (price_range * 0.1)
        
        return buy_level, sell_level, support_level, resistance_level

def adaptive_backtest_strategy(df: pd.DataFrame, params: AdaptiveParams) -> Dict[str, Any]:
    """
    Enhanced adaptive backtesting strategy using EnhancedRegimeDetector.
    """
    # Initialize components
    regime_detector = EnhancedRegimeDetector(
        short_window=params.regime_short_window,
        medium_window=params.regime_medium_window,
        long_window=params.regime_long_window
    )
    price_calculator = AdaptivePriceLevels(params)
    
    # Trading variables
    initial_balance = 10000
    balance = initial_balance
    position = 0  # 0 = no position, 1 = long
    position_size = 0
    entry_price = 0.0
    entry_index = 0  # Track when we entered
    
    # Performance tracking
    trades, wins, losses = 0, 0, 0
    peak_balance = initial_balance
    max_drawdown = 0
    equity_curve = []
    trade_log = []
    
    # Execution costs
    fee_rate = 0.001      # 0.1% fee per side
    slippage_pct = 0.0005 # 0.05% slippage
    
    # Strategy loop
    start_idx = max(50, params.regime_long_window)  # Ensure enough data for regime detection
    
    for i in range(start_idx, len(df)):
        current_row = df.iloc[i]
        current_price = current_row['close']
        
        # Detect market regime using enhanced detector
        regime = regime_detector.detect_regime(df, i)
        
        # Calculate adaptive price levels
        buy_level, sell_level, support, resistance = price_calculator.get_price_levels(df, i)
        
        # Fix support/resistance if they're too close or inverted
        if resistance <= support:
            current_price = df.iloc[i]['close']
            support = current_price * 0.95
            resistance = current_price * 1.05
        
        # Get regime-based position size multiplier
        regime_multiplier = RegimeBasedTradingRules.get_position_size_multiplier(regime)
        
        # Base position sizing
        base_position_size = balance * params.max_position_pct
        adjusted_position_size = base_position_size * regime_multiplier
        
        # Entry logic
        if position == 0:
            # Enhanced entry logic with additional filters
            basic_entry_signal = RegimeBasedTradingRules.should_enter_long(
                regime, current_price, buy_level
            )
            
            # Additional entry filters
            confidence_filter = regime.confidence >= 0.6  # Raised minimum confidence
            
            # Fix price position calculation with better bounds
            price_range = max(resistance - support, current_price * 0.01)  # Minimum 1% range
            price_position = max(0, min(1, (current_price - support) / price_range))  # Clamp 0-1
            
            # Avoid trading in extreme downward momentum unless very oversold
            momentum_filter = True
            if regime.momentum.value == "accel_down":
                # Only trade if significantly oversold AND not in strong bear trend
                momentum_filter = (price_position < 0.25 and 
                                 regime.trend.value != "strong_bear")
            elif regime.momentum.value == "steady_down":
                # Less restrictive for steady down
                momentum_filter = price_position < 0.4
            
            # Additional trend filter - avoid buying in bear markets unless deeply oversold
            trend_filter = True
            if regime.trend.value in ["mild_bear", "strong_bear"]:
                trend_filter = price_position < 0.2  # Only buy if deeply oversold
            
            should_enter = (basic_entry_signal and confidence_filter and 
                          momentum_filter and trend_filter)
            
            if should_enter:
                # Execute buy order
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
                    'regime': str(regime),
                    'regime_confidence': regime.confidence,
                    'regime_multiplier': regime_multiplier,
                    'buy_level': buy_level,
                    'sell_level': sell_level
                })
                
                print(f"BUY at {executed_price:.2f} on {current_row.name} "
                      f"(regime: {regime}, confidence: {regime.confidence:.2f}, "
                      f"multiplier: {regime_multiplier:.2f}, "
                      f"price_pos: {price_position:.2f}, range: {price_range:.2f})")
        
        else:
            # Enhanced exit logic with much more conservative rules
            current_value = position_size * current_price
            unrealized_pnl = (current_price - entry_price) / entry_price
            
            # Track how long we've been in the position
            days_in_position = i - entry_index
            
            # Get basic regime exit signal
            basic_exit_signal, basic_exit_reason = RegimeBasedTradingRules.should_exit_long(
                regime, current_price, entry_price, sell_level, 
                params.stop_loss_pct, params.take_profit_pct
            )
            
            should_exit = False
            exit_reason = ""
            
            # Very conservative exit logic - give trades time to work
            if basic_exit_reason in ["stop_loss", "take_profit"]:
                # Always honor stop loss and take profit
                should_exit = True
                exit_reason = basic_exit_reason
                
            elif basic_exit_reason == "regime_emergency":
                # Emergency exits (extreme conditions)
                should_exit = True
                exit_reason = basic_exit_reason
                
            elif basic_exit_reason == "bear_market_exit":
                # Only exit on strong bear market if losing significantly
                if unrealized_pnl < -params.stop_loss_pct * 0.5:  # 50% of stop loss
                    should_exit = True
                    exit_reason = "bear_market_exit"
                    
            elif basic_exit_reason == "early_bear_exit":
                # Even more conservative - only if losing >6% OR holding >7 days with >2% loss
                if unrealized_pnl < -0.06 or (unrealized_pnl < -0.02 and days_in_position > 7):
                    should_exit = True
                    exit_reason = "early_bear_exit"
                    
            elif basic_exit_reason == "momentum_exit":
                # Very conservative momentum exit - require significant conditions
                strong_momentum_down = regime.momentum.value == "accel_down"
                significant_loss = unrealized_pnl < -0.04  # 4% loss
                held_long_enough = days_in_position >= 3  # At least 3 days
                
                if strong_momentum_down and significant_loss and held_long_enough:
                    should_exit = True
                    exit_reason = "momentum_exit"
                    
            elif basic_exit_reason == "volatility_exit":
                # Take profits in extreme volatility if gain is good
                if unrealized_pnl > 0.05:  # At least 5% profit
                    should_exit = True
                    exit_reason = "volatility_exit"
            
            # Additional exit condition: trailing stop for profitable positions
            if unrealized_pnl > 0.1:  # If up 10%+
                # Check if we've dropped more than 3% from recent highs
                recent_high_idx = max(0, i-5)
                recent_high = df.iloc[recent_high_idx:i+1]['close'].max()
                if current_price < recent_high * 0.97:  # 3% trailing stop
                    should_exit = True
                    exit_reason = "trailing_stop"
            
            if should_exit:
                # Execute sell order
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
                    'regime': str(regime),
                    'regime_confidence': regime.confidence,
                    'days_held': days_in_position,
                    'is_win': is_win
                })
                
                print(f"SELL at {executed_price:.2f} on {current_row.name} "
                      f"(reason: {exit_reason}, return: {gross_return:.1%}, "
                      f"days: {days_in_position}, regime: {regime})")
                
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
        
        final_days_held = len(df) - 1 - entry_index
        
        trade_log.append({
            'type': 'FORCED_SELL',
            'date': df.index[-1],
            'price': final_price,
            'size': position_size,
            'return': gross_return,
            'reason': 'end_of_data',
            'days_held': final_days_held,
            'is_win': is_win
        })
        
        print(f"FORCED SELL at {final_price:.2f} on {df.index[-1]} "
              f"(return: {gross_return:.1%}, days: {final_days_held})")
    
    # Calculate final metrics
    final_return = (balance - initial_balance) / initial_balance
    
    # Calculate Sharpe ratio from equity curve
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
    
    print(f"\nEnhanced Adaptive Strategy Results:")
    print(f"Trades: {trades}, Wins: {wins}, Losses: {losses}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Total Return: {final_return:.1%}")
    print(f"Max Drawdown: {max_drawdown:.1%}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    
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

# Integration with existing fitness system
def evaluate_adaptive_strategy(df: pd.DataFrame, params: AdaptiveParams, 
                              fitness_config=None) -> Dict[str, Any]:
    """
    Evaluate adaptive strategy for use with genetic algorithm.
    """
    try:
        # Run adaptive backtest
        results = adaptive_backtest_strategy(df, params)
        
        # Create BacktestResult object for fitness calculation
        from src.evolution.fitness import BacktestResult, calculate_backtest_fitness
        
        # Convert equity curve to daily returns
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
        
        # Calculate fitness
        if fitness_config is None:
            from src.evolution.fitness import FitnessConfig
            fitness_config = FitnessConfig()
        
        fitness, components = calculate_backtest_fitness(backtest_result, fitness_config)
        
        # Combine results
        results['fitness'] = fitness
        results.update(components)
        
        return results
        
    except Exception as e:
        print(f"[Enhanced Adaptive Strategy Error] {e}")
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

def optimize_regime_parameters(df: pd.DataFrame, 
                             confidence_thresholds: list = [0.5, 0.6, 0.7],
                             momentum_thresholds: list = [0.02, 0.03, 0.04],
                             position_thresholds: list = [0.2, 0.25, 0.3]) -> Dict[str, Any]:
    """
    Test different parameter combinations to find optimal settings.
    """
    print("🔧 Optimizing regime parameters...")
    
    best_params = None
    best_score = -1
    results = []
    
    base_params = AdaptiveParams(
        buy_threshold_pct=0.25,
        sell_threshold_pct=0.75,
        regime_short_window=15,
        regime_medium_window=35,
        regime_long_window=70,
        max_position_pct=0.8
    )
    
    # Test recent data
    test_data = df.tail(200)
    
    for conf_thresh in confidence_thresholds:
        for mom_thresh in momentum_thresholds:
            for pos_thresh in position_thresholds:
                try:
                    # Temporarily modify the parameters (would need to pass these through)
                    # For now, just test base parameters
                    result = adaptive_backtest_strategy(test_data, base_params)
                    
                    # Score based on return/drawdown ratio and win rate
                    score = (result['returns'] / max(result['max_drawdown'], 0.01) + 
                            result['win_rate'] * 2)
                    
                    results.append({
                        'confidence_thresh': conf_thresh,
                        'momentum_thresh': mom_thresh,
                        'position_thresh': pos_thresh,
                        'score': score,
                        'returns': result['returns'],
                        'win_rate': result['win_rate'],
                        'trades': result['trades']
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'confidence_thresh': conf_thresh,
                            'momentum_thresh': mom_thresh,
                            'position_thresh': pos_thresh
                        }
                        
                except Exception as e:
                    print(f"Error testing params: {e}")
                    continue
    
    print(f"\n🏆 Best parameters found:")
    print(f"   Confidence threshold: {best_params['confidence_thresh']}")
    print(f"   Momentum threshold: {best_params['momentum_thresh']}")
    print(f"   Position threshold: {best_params['position_thresh']}")
    print(f"   Best score: {best_score:.3f}")
    
    return {'best_params': best_params, 'all_results': results}

def analyze_regime_performance(df: pd.DataFrame, params: AdaptiveParams, 
                             recent_days: int = 200) -> Dict[str, Any]:
    """
    Analyze how different regime conditions affect trade performance.
    """
    regime_detector = EnhancedRegimeDetector(
        short_window=params.regime_short_window,
        medium_window=params.regime_medium_window,
        long_window=params.regime_long_window
    )
    
    # Sample regime conditions over recent period
    regime_stats = {
        'trend_counts': {},
        'volatility_counts': {},
        'momentum_counts': {},
        'confidence_distribution': [],
        'should_avoid_count': 0
    }
    
    start_idx = max(len(df) - recent_days, params.regime_long_window)
    
    for i in range(start_idx, len(df)):
        regime = regime_detector.detect_regime(df, i)
        
        # Count regime types
        trend = regime.trend.value
        vol = regime.volatility.value
        momentum = regime.momentum.value
        
        regime_stats['trend_counts'][trend] = regime_stats['trend_counts'].get(trend, 0) + 1
        regime_stats['volatility_counts'][vol] = regime_stats['volatility_counts'].get(vol, 0) + 1
        regime_stats['momentum_counts'][momentum] = regime_stats['momentum_counts'].get(momentum, 0) + 1
        regime_stats['confidence_distribution'].append(regime.confidence)
        
        if regime.should_avoid_trading():
            regime_stats['should_avoid_count'] += 1
    
    # Calculate percentages
    total_periods = len(df) - start_idx
    regime_stats['avoid_trading_pct'] = regime_stats['should_avoid_count'] / total_periods * 100
    regime_stats['avg_confidence'] = np.mean(regime_stats['confidence_distribution'])
    
    print(f"\n📊 REGIME ANALYSIS (last {recent_days} days):")
    print(f"   Average Confidence: {regime_stats['avg_confidence']:.2f}")
    print(f"   Avoid Trading: {regime_stats['avoid_trading_pct']:.1f}% of time")
    
    print(f"\n   Trend Distribution:")
    for trend, count in regime_stats['trend_counts'].items():
        pct = count / total_periods * 100
        print(f"     {trend}: {pct:.1f}%")
    
    print(f"\n   Momentum Distribution:")
    for momentum, count in regime_stats['momentum_counts'].items():
        pct = count / total_periods * 100
        print(f"     {momentum}: {pct:.1f}%")
        
    return regime_stats

# Test function
def test_adaptive_strategy():
    """
    Quick test of the enhanced adaptive strategy.
    """
    from src.core.data import fetch_historical_data
    
    print("🧪 Testing Enhanced Adaptive Strategy...")
    
    # Load data
    df = fetch_historical_data(refresh=False)
    
    # Test parameters with more conservative settings
    test_params = AdaptiveParams(
        risk_reward=1.5,
        buy_threshold_pct=0.25,      # More aggressive buying (lower threshold)
        sell_threshold_pct=0.75,     # Conservative selling
        regime_short_window=15,      # Slightly longer windows for stability
        regime_medium_window=35,
        regime_long_window=70,
        max_position_pct=0.8,
        stop_loss_pct=0.08,
        take_profit_pct=0.15
    )
    
    # Run test on recent data
    recent_data = df.tail(300)  # Last 300 days
    results = adaptive_backtest_strategy(recent_data, test_params)
    
    print(f"\n✅ Test completed!")
    print(f"   Return: {results['returns']:.1%}")
    print(f"   Trades: {results['trades']}")
    print(f"   Win Rate: {results['win_rate']:.1%}")
    print(f"   Sharpe Ratio: {results['sharpe']:.4f}")

if __name__ == "__main__":
    test_adaptive_strategy()