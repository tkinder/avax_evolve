# solana_backtester.py
"""
Solana-specific backtester using AVAX's proven validation approach
Adapted from AVAX's sophisticated backtest system
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from src.evolution.fitness import BacktestResult, calculate_backtest_fitness, FitnessConfig

@dataclass
class SolanaParams:
    """
    Solana-specific strategy parameters adapted from AVAX AdaptiveParams
    """
    # Core strategy parameters (from DEAP optimization)
    risk_reward: float = 1.5
    trend_strength: float = 1.0
    entry_threshold: float = 0.5
    confidence: float = 1.0
    
    # Solana-specific thresholds (optimized by DEAP)
    buy_threshold_pct: float = 0.561     # Solana's discovered optimal: 56.1%
    sell_threshold_pct: float = 0.745    # Solana's discovered optimal: 74.5%
    
    # Regime detection parameters
    regime_short_window: int = 10
    regime_medium_window: int = 30
    regime_long_window: int = 60
    
    # Position sizing multipliers (from DEAP)
    bull_multiplier: float = 1.14
    bear_multiplier: float = 0.70
    high_vol_multiplier: float = 0.82
    low_vol_multiplier: float = 1.11
    
    # Risk management (Solana-specific from DEAP)
    max_position_pct: float = 0.591      # Solana more conservative position size
    stop_loss_pct: float = 0.376         # Solana's high 37.6% stop (contrarian!)
    take_profit_pct: float = 0.120       # Solana's 12.0% target
    
    # Price level calculation
    price_lookback_days: int = 50

class SolanaPriceLevels:
    """
    Calculate Solana-specific adaptive buy/sell levels
    Adapted from AVAX's AdaptivePriceLevels
    """
    
    def __init__(self, params: SolanaParams):
        self.params = params
    
    def get_price_levels(self, df: pd.DataFrame, current_idx: int, 
                        lookback_days: int = None) -> Tuple[float, float, float, float]:
        """
        Calculate Solana-specific buy/sell levels based on recent price range.
        """
        if lookback_days is None:
            lookback_days = self.params.price_lookback_days
            
        if current_idx < lookback_days:
            lookback_days = current_idx
        
        if lookback_days < 10:
            # Fallback for insufficient data
            current_price = df.iloc[current_idx]['close']
            return current_price * 0.90, current_price * 1.15, current_price * 0.85, current_price * 1.20
        
        start_idx = max(0, current_idx - lookback_days)
        recent_data = df.iloc[start_idx:current_idx + 1]
        
        # Calculate price range
        price_low = recent_data['close'].min()
        price_high = recent_data['close'].max()
        price_range = price_high - price_low
        
        # Solana-specific levels using DEAP-optimized thresholds
        buy_level = price_low + (price_range * self.params.buy_threshold_pct)
        sell_level = price_low + (price_range * self.params.sell_threshold_pct)
        
        # Solana support/resistance (balanced between AVAX and Bitcoin)
        support_level = min(recent_data['close'].quantile(0.20), price_low * 1.02)  # Moderate for Solana
        resistance_level = max(recent_data['close'].quantile(0.80), price_high * 0.98)
        
        # Ensure minimum spread
        if resistance_level - support_level < price_range * 0.08:  # Moderate spread for Solana
            mid_price = (support_level + resistance_level) / 2
            support_level = mid_price - (price_range * 0.08)
            resistance_level = mid_price + (price_range * 0.08)
        
        return buy_level, sell_level, support_level, resistance_level

def solana_backtest_strategy(df: pd.DataFrame, params: SolanaParams) -> Dict[str, Any]:
    """
    Solana-specific backtesting strategy using AVAX's proven approach
    """
    # Initialize Solana price calculator
    price_calculator = SolanaPriceLevels(params)
    
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
    
    # Solana-specific execution costs (between AVAX and Bitcoin)
    fee_rate = 0.001       # 0.1% fee (Solana moderate fees)
    slippage_pct = 0.0005  # 0.05% slippage (Solana decent liquidity)
    
    # Strategy loop (ensure enough data)
    start_idx = max(50, params.regime_long_window)
    
    # Add simple regime detection for Solana
    df['sma_short'] = df['close'].rolling(window=params.regime_short_window).mean()
    df['sma_medium'] = df['close'].rolling(window=params.regime_medium_window).mean()
    df['sma_long'] = df['close'].rolling(window=params.regime_long_window).mean()
    
    for i in range(start_idx, len(df)):
        current_row = df.iloc[i]
        current_price = current_row['close']
        
        # Solana regime detection (simplified but effective)
        sma_short = current_row.get('sma_short', current_price)
        sma_medium = current_row.get('sma_medium', current_price)
        sma_long = current_row.get('sma_long', current_price)
        
        # Determine Solana market regime
        if pd.isna(sma_long):
            continue
            
        # Solana trend classification (moderate sensitivity)
        if current_price > sma_long * 1.025 and sma_short > sma_medium:  # 2.5% buffer for Solana
            solana_trend = "bullish"
            regime_multiplier = params.bull_multiplier
        elif current_price < sma_long * 0.975 and sma_short < sma_medium:
            solana_trend = "bearish"
            regime_multiplier = params.bear_multiplier
        else:
            solana_trend = "neutral"
            regime_multiplier = 1.0
        
        # Solana volatility assessment (last 14 days)
        volume_strength = 1.0  # Default volume strength
        if i >= 14:
            recent_returns = df.iloc[i-14:i]['close'].pct_change().dropna()
            volatility = recent_returns.std()
            if volatility > 0.06:  # High volatility for Solana
                regime_multiplier *= params.high_vol_multiplier
            elif volatility < 0.025:  # Low volatility for Solana
                regime_multiplier *= params.low_vol_multiplier
            
            # Solana volume assessment (if volume column exists)
            if 'volume' in df.columns:
                recent_volume = df.iloc[i-14:i]['volume'].mean()
                current_volume = current_row.get('volume', recent_volume)
                volume_strength = min(2.0, max(0.5, current_volume / recent_volume))
        
        # Calculate Solana-specific price levels
        buy_level, sell_level, support, resistance = price_calculator.get_price_levels(df, i)
        
        # Fix support/resistance if needed
        if resistance <= support:
            support = current_price * 0.95
            resistance = current_price * 1.05
        
        # Position sizing
        base_position_size = balance * params.max_position_pct
        adjusted_position_size = base_position_size * regime_multiplier
        
        # ENTRY LOGIC (Solana-specific)
        if position == 0:
            # Solana entry conditions (moderate selectivity 56.1%)
            price_range = max(resistance - support, current_price * 0.02)
            price_position = max(0, min(1, (current_price - support) / price_range))
            
            # Solana entry: moderate selectivity (56.1% means wait for moderate oversold)
            solana_entry_signal = price_position < (1 - params.buy_threshold_pct)  # Inverted threshold
            
            # Solana-specific filters (balanced approach)
            trend_ok = solana_trend != "bearish" or price_position < 0.2  # Avoid bearish unless deeply oversold
            
            # Solana volume filter (require decent volume for entry)
            volume_ok = volume_strength >= 0.8  # Reasonable volume required
            
            # Solana confidence filter (moderate requirements)
            confidence_ok = True
            if solana_trend == "neutral":
                confidence_ok = price_position < 0.4  # Moderate oversold in neutral
            elif solana_trend == "bearish":
                confidence_ok = price_position < 0.25  # Very oversold in bearish
            
            should_enter = solana_entry_signal and trend_ok and volume_ok and confidence_ok
            
            if should_enter:
                # Execute Solana buy
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
                    'trend': solana_trend,
                    'regime_multiplier': regime_multiplier,
                    'volume_strength': volume_strength,
                    'price_position': price_position,
                    'buy_level': buy_level,
                    'sell_level': sell_level
                })
                
                print(f"🟢 SOL BUY at ${executed_price:,.2f} on {current_row.name} "
                      f"(trend: {solana_trend}, pos: {price_position:.2f}, vol: {volume_strength:.2f})")
        
        else:  # In position
            # EXIT LOGIC (Solana-specific - contrarian with high stop loss)
            current_value = position_size * current_price
            unrealized_pnl = (current_price - entry_price) / entry_price
            days_in_position = i - entry_index
            
            should_exit = False
            exit_reason = ""
            
            # Solana stop loss (very high 37.6% - contrarian approach!)
            if unrealized_pnl <= -params.stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Solana take profit (12.0% target)
            elif unrealized_pnl >= params.take_profit_pct:
                should_exit = True
                exit_reason = "take_profit"
            
            # Solana sell threshold (74.5% - wait for high peaks)
            else:
                price_range = max(resistance - support, current_price * 0.02)
                price_position = max(0, min(1, (current_price - support) / price_range))
                
                if price_position > params.sell_threshold_pct:  # 74.5% threshold (high exit)
                    should_exit = True
                    exit_reason = "sell_threshold"
            
            # Solana emergency exits (rare due to high stop loss)
            if solana_trend == "bearish" and unrealized_pnl < -0.15 and days_in_position > 5:
                should_exit = True
                exit_reason = "bear_emergency"
            
            # Solana momentum exit (high volatility asset)
            if i >= 7:  # Look at recent momentum
                recent_returns = df.iloc[i-7:i]['close'].pct_change().dropna()
                if len(recent_returns) > 0:
                    momentum = recent_returns.mean()
                    if momentum < -0.03 and unrealized_pnl < -0.10:  # Strong down momentum + loss
                        should_exit = True
                        exit_reason = "momentum_exit"
            
            # Solana trailing stop for big gains (wider than Bitcoin)
            if unrealized_pnl > 0.20:  # 20%+ profit
                recent_high_idx = max(0, i-5)  # Last 5 days for Solana
                recent_high = df.iloc[recent_high_idx:i+1]['close'].max()
                if current_price < recent_high * 0.95:  # 5% trailing for Solana
                    should_exit = True
                    exit_reason = "trailing_stop"
            
            # Solana volume-based exit (low volume on gains)
            if unrealized_pnl > 0.08 and volume_strength < 0.6:  # Low volume on 8%+ gains
                should_exit = True
                exit_reason = "volume_exit"
            
            if should_exit:
                # Execute Solana sell
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
                    'trend': solana_trend,
                    'days_held': days_in_position,
                    'volume_strength': volume_strength,
                    'is_win': is_win
                })
                
                print(f"🔴 SOL SELL at ${executed_price:,.2f} on {current_row.name} "
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
    
    print(f"\n🌟 Solana Strategy Results:")
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

def evaluate_solana_strategy_with_backtest(df: pd.DataFrame, params: SolanaParams, 
                                         fitness_config=None) -> Dict[str, Any]:
    """
    Evaluate Solana strategy using AVAX's proven backtest validation approach
    """
    try:
        # Run Solana backtest
        results = solana_backtest_strategy(df, params)
        
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
                # Solana-specific fitness config (balanced)
                min_trades=2,                    # Solana contrarian - fewer trades ok
                max_drawdown_threshold=0.45,     # Higher drawdown tolerance 
                min_profit_threshold=-0.15,      # Allow moderate losses
                profitability_weight=0.40,       # Focus on profits
                risk_adjusted_weight=0.25,       # Risk-adjusted returns
                drawdown_weight=0.25,            # Drawdown control
                trade_quality_weight=0.10        # Trade quality
            )
        
        fitness, components = calculate_backtest_fitness(backtest_result, fitness_config)
        
        # Combine results (AVAX style)
        results['fitness'] = fitness
        results.update(components)
        
        return results
        
    except Exception as e:
        print(f"[Solana Strategy Error] {e}")
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
def test_solana_backtester():
    """
    Test the enhanced Solana backtester
    """
    from solana_data_pipeline import SolanaDataPipeline
    
    print("🧪 Testing Enhanced Solana Backtester...")
    
    # Get Solana data
    pipeline = SolanaDataPipeline()
    df = pipeline.get_complete_dataset(historical_days=90)
    
    if df is None:
        print("❌ No Solana data available")
        return
    
    # Test with DEAP-optimized parameters
    test_params = SolanaParams(
        buy_threshold_pct=0.561,     # From DEAP optimization
        sell_threshold_pct=0.745,    # From DEAP optimization
        stop_loss_pct=0.376,         # From DEAP optimization (high!)
        take_profit_pct=0.120,       # From DEAP optimization
        max_position_pct=0.591,      # From DEAP optimization
        bull_multiplier=1.14,        # From DEAP optimization
        bear_multiplier=0.70,        # From DEAP optimization
        high_vol_multiplier=0.82,    # From DEAP optimization
        low_vol_multiplier=1.11      # From DEAP optimization
    )
    
    # Run enhanced backtest
    results = evaluate_solana_strategy_with_backtest(df, test_params)
    
    print(f"\n✅ Enhanced Solana Backtester Results:")
    print(f"   Fitness Score: {results['fitness']:.4f}")
    print(f"   Return: {results['returns']:.1%}")
    print(f"   Trades: {results['trades']}")
    print(f"   Win Rate: {results.get('win_rate', 0):.1%}")
    print(f"   Max Drawdown: {results['max_drawdown']:.1%}")

if __name__ == "__main__":
    test_solana_backtester()
