#!/usr/bin/env python3
"""
Bitcoin Trend-Following + Regime-Aware Strategy
Adapts strategy type based on market regime: trend-following in bulls, mean-reversion in bears
"""

import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional

# Import AVAX's regime detection
from src.core.regime_detector import (
    EnhancedRegimeDetector, RegimeBasedTradingRules, 
    TrendRegime, VolatilityRegime, MomentumRegime
)

@dataclass
class BitcoinTrendRegimeParams:
    """Bitcoin trend-following + regime-aware parameters"""
    
    # TREND-FOLLOWING PARAMETERS (for bull markets)
    trend_breakout_pct: float = 0.65        # Buy when price breaks above 65% of range
    trend_stop_loss_pct: float = 0.08       # Wider stops for trend following (8%)
    trend_position_size: float = 1.2        # Larger positions in trends
    trend_hold_periods: int = 20            # Hold longer in trends
    
    # MEAN-REVERSION PARAMETERS (for bear/sideways markets)
    reversion_buy_pct: float = 0.25         # Buy dips at 25% of range
    reversion_sell_pct: float = 0.60        # Quick exits at 60% of range
    reversion_stop_loss_pct: float = 0.04   # Tight stops for mean reversion (4%)
    reversion_position_size: float = 0.8    # Smaller positions in mean reversion
    
    # REGIME PARAMETERS
    bull_confidence_threshold: float = 0.4   # Min confidence for bull strategies
    bear_confidence_threshold: float = 0.5   # Higher confidence needed for bear trades
    sideways_avoid_threshold: float = 0.3    # Avoid trading if sideways confidence > 30%
    
    # RISK MANAGEMENT
    max_position_size: float = 1.5          # Maximum position size
    max_trades_per_regime: int = 5          # Limit trades per regime to avoid overtrading
    regime_change_exit: bool = True         # Exit positions on major regime changes
    
    # REGIME WINDOWS
    regime_short_window: int = 10
    regime_medium_window: int = 30
    regime_long_window: int = 60
    
    # VOLATILITY ADJUSTMENTS
    high_vol_reduction: float = 0.6         # Reduce positions 40% in high volatility
    low_vol_boost: float = 1.3              # Boost positions 30% in low volatility

class TrendRegimeStrategy:
    """Adaptive strategy that switches between trend-following and mean-reversion based on regime"""
    
    def __init__(self, params: BitcoinTrendRegimeParams):
        self.params = params
        self.regime_detector = EnhancedRegimeDetector(
            short_window=params.regime_short_window,
            medium_window=params.regime_medium_window,
            long_window=params.regime_long_window
        )
        self.regime_trade_count = {}  # Track trades per regime
        self.current_strategy = None  # Track current strategy type
        
    def should_enter_position(self, regime, current_price, price_range_data) -> Tuple[bool, str, float]:
        """
        Determine if we should enter a position based on regime-adaptive strategy
        Returns: (should_enter, strategy_type, position_size)
        """
        price_min, price_max = price_range_data['min'], price_range_data['max']
        price_range = price_max - price_min
        price_percentile = (current_price - price_min) / price_range
        
        # Check regime trade limits
        regime_key = f"{regime.trend.value}_{regime.volatility.value}"
        regime_trades = self.regime_trade_count.get(regime_key, 0)
        if regime_trades >= self.params.max_trades_per_regime:
            return False, "max_trades_reached", 0
        
        # BULL MARKET: Trend-following strategy
        if regime.is_bullish() and regime.confidence >= self.params.bull_confidence_threshold:
            # Buy on breakouts above resistance
            breakout_level = self.params.trend_breakout_pct
            
            if price_percentile >= breakout_level:
                # Confirm upward momentum
                if regime.momentum in [MomentumRegime.ACCELERATING_UP, MomentumRegime.STEADY_UP]:
                    position_size = self.params.trend_position_size
                    
                    # Adjust for volatility
                    if regime.is_high_volatility():
                        position_size *= self.params.high_vol_reduction
                    else:
                        position_size *= self.params.low_vol_boost
                    
                    position_size = min(position_size, self.params.max_position_size)
                    return True, "trend_following", position_size
        
        # BEAR MARKET: Mean-reversion strategy
        elif regime.is_bearish() and regime.confidence >= self.params.bear_confidence_threshold:
            # Buy dips at oversold levels
            dip_level = self.params.reversion_buy_pct
            
            if price_percentile <= dip_level:
                # Look for oversold bounce potential
                if regime.momentum in [MomentumRegime.STALLING, MomentumRegime.STEADY_DOWN]:
                    position_size = self.params.reversion_position_size
                    
                    # More conservative in bear markets
                    if regime.trend == TrendRegime.STRONG_BEAR:
                        position_size *= 0.5
                    
                    # Adjust for volatility
                    if regime.is_high_volatility():
                        position_size *= self.params.high_vol_reduction
                    
                    position_size = min(position_size, self.params.max_position_size * 0.8)  # Lower max in bear
                    return True, "mean_reversion", position_size
        
        # SIDEWAYS MARKET: Selective mean-reversion
        elif regime.trend == TrendRegime.SIDEWAYS and regime.confidence >= self.params.bull_confidence_threshold:
            # Only trade in low volatility sideways markets
            if regime.volatility in [VolatilityRegime.LOW, VolatilityRegime.NORMAL]:
                # Buy at bottom of range, sell at top
                if price_percentile <= self.params.reversion_buy_pct:
                    position_size = self.params.reversion_position_size * 0.7  # Even smaller in sideways
                    return True, "sideways_reversion", position_size
        
        return False, "no_signal", 0
    
    def should_exit_position(self, regime, current_price, entry_data) -> Tuple[bool, str]:
        """
        Determine exit based on strategy type and regime
        Returns: (should_exit, exit_reason)
        """
        entry_price = entry_data['price']
        strategy_type = entry_data['strategy']
        entry_regime = entry_data.get('regime', regime)
        unrealized_return = (current_price - entry_price) / entry_price
        
        # Emergency exits for regime changes
        if self.params.regime_change_exit:
            # Exit trend-following positions if regime turns bearish
            if strategy_type == "trend_following" and regime.is_bearish():
                return True, "regime_change_bearish"
            
            # Exit mean-reversion positions if strong trend emerges
            if strategy_type in ["mean_reversion", "sideways_reversion"] and regime.trend == TrendRegime.STRONG_BULL:
                return True, "regime_change_bullish"
        
        # Strategy-specific exits
        if strategy_type == "trend_following":
            # Trend-following exits: ride the trend, wider stops
            if unrealized_return <= -self.params.trend_stop_loss_pct:
                return True, "trend_stop_loss"
            
            # Exit if momentum turns negative
            if regime.momentum in [MomentumRegime.ACCELERATING_DOWN, MomentumRegime.STEADY_DOWN]:
                return True, "momentum_reversal"
            
            # Take profits on extreme overbought + momentum stalling
            if unrealized_return > 0.15 and regime.momentum == MomentumRegime.STALLING:
                return True, "trend_profit_momentum_stall"
        
        elif strategy_type in ["mean_reversion", "sideways_reversion"]:
            # Mean-reversion exits: quick profits, tight stops
            if unrealized_return <= -self.params.reversion_stop_loss_pct:
                return True, "reversion_stop_loss"
            
            # Quick profit taking in mean reversion
            price_range_data = entry_data.get('price_range_data', {})
            if price_range_data:
                price_min, price_max = price_range_data['min'], price_range_data['max']
                price_percentile = (current_price - price_min) / (price_max - price_min)
                
                if price_percentile >= self.params.reversion_sell_pct:
                    return True, "reversion_profit_target"
        
        # Universal exits
        if regime.volatility == VolatilityRegime.EXTREME and unrealized_return > 0:
            return True, "extreme_volatility_exit"
        
        if regime.should_avoid_trading():
            return True, "regime_risk_exit"
        
        return False, "hold"

def evaluate_bitcoin_trend_regime_strategy(df: pd.DataFrame, 
                                          params: BitcoinTrendRegimeParams) -> dict:
    """
    Evaluate Bitcoin strategy with trend-following + regime awareness
    """
    try:
        print(f"🚀 Running trend-following + regime-aware Bitcoin backtest...")
        
        strategy = TrendRegimeStrategy(params)
        
        # Setup
        initial_balance = 10000.0
        balance = initial_balance
        position = 0.0
        entry_data = {}
        trades, wins, losses = 0, 0, 0
        peak_balance = initial_balance
        max_drawdown = 0.0
        
        # Calculate price range data
        price_min = df['close'].min()
        price_max = df['close'].max()
        price_range_data = {'min': price_min, 'max': price_max}
        
        trade_log = []
        regime_log = []
        balance_history = [initial_balance]
        strategy_performance = {'trend_following': [], 'mean_reversion': [], 'sideways_reversion': []}
        
        print(f"   Price range: ${price_min:,.0f} - ${price_max:,.0f}")
        print(f"   Strategy: Trend-following in bulls, mean-reversion in bears/sideways")
        
        for i in range(params.regime_long_window, len(df)):
            current_price = df['close'].iloc[i]
            current_date = df.index[i]
            
            # Detect current market regime
            regime = strategy.regime_detector.detect_regime(df, i)
            regime_log.append({
                'date': current_date,
                'regime': str(regime),
                'trend': regime.trend.value,
                'volatility': regime.volatility.value,
                'momentum': regime.momentum.value,
                'confidence': regime.confidence,
                'price': current_price
            })
            
            # ENTRY LOGIC
            if position == 0:
                should_enter, strategy_type, position_size = strategy.should_enter_position(
                    regime, current_price, price_range_data
                )
                
                if should_enter:
                    entry_price = current_price * 1.001  # Small slippage
                    position = position_size
                    trades += 1
                    
                    # Track regime trades
                    regime_key = f"{regime.trend.value}_{regime.volatility.value}"
                    strategy.regime_trade_count[regime_key] = strategy.regime_trade_count.get(regime_key, 0) + 1
                    
                    entry_data = {
                        'price': entry_price,
                        'strategy': strategy_type,
                        'regime': regime,
                        'date': current_date,
                        'price_range_data': price_range_data
                    }
                    
                    trade_log.append({
                        'type': 'BUY',
                        'date': current_date,
                        'price': entry_price,
                        'position_size': position_size,
                        'strategy': strategy_type,
                        'regime': str(regime),
                        'confidence': regime.confidence,
                        'balance_before': balance
                    })
                    
                    if len(trade_log) <= 10:  # Print first 10 trades
                        print(f"   BUY: ${entry_price:,.0f} on {current_date.strftime('%Y-%m-%d')} "
                              f"({strategy_type}, {regime.trend.value}, pos: {position_size:.1f}x)")
            
            # EXIT LOGIC
            else:
                should_exit, exit_reason = strategy.should_exit_position(
                    regime, current_price, entry_data
                )
                
                if should_exit:
                    exit_price = current_price * 0.999  # Small slippage
                    gross_return = (exit_price - entry_data['price']) / entry_data['price']
                    net_return = gross_return - 0.002  # Fees
                    
                    # Calculate position return
                    position_return = net_return * position
                    balance *= (1 + position_return)
                    
                    is_win = net_return > 0
                    wins += 1 if is_win else 0
                    losses += 1 if not is_win else 0
                    
                    # Track strategy performance
                    strategy_type = entry_data['strategy']
                    strategy_performance[strategy_type].append(net_return)
                    
                    trade_log.append({
                        'type': 'SELL',
                        'date': current_date,
                        'price': exit_price,
                        'gross_return': gross_return,
                        'net_return': net_return,
                        'position_return': position_return,
                        'reason': exit_reason,
                        'strategy': strategy_type,
                        'regime': str(regime),
                        'balance_after': balance,
                        'hold_days': (current_date - entry_data['date']).days
                    })
                    
                    if len([t for t in trade_log if t['type'] == 'SELL']) <= 10:  # Print first 10 exits
                        hold_days = (current_date - entry_data['date']).days
                        print(f"   SELL: ${exit_price:,.0f} on {current_date.strftime('%Y-%m-%d')} "
                              f"({net_return:+.1%}, {exit_reason}, {hold_days}d hold)")
                    
                    position = 0
                    entry_data = {}
            
            # Track balance and drawdown
            balance_history.append(balance)
            peak_balance = max(peak_balance, balance)
            current_drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, current_drawdown)
        
        # Force exit if still in position
        if position > 0:
            final_price = df['close'].iloc[-1] * 0.999
            gross_return = (final_price - entry_data['price']) / entry_data['price']
            net_return = gross_return - 0.002
            position_return = net_return * position
            balance *= (1 + position_return)
            
            is_win = net_return > 0
            wins += 1 if is_win else 0
            losses += 1 if not is_win else 0
            
            print(f"   FORCED EXIT: ${final_price:,.0f} ({net_return:+.1%})")
        
        # Calculate comprehensive metrics
        total_return = (balance - initial_balance) / initial_balance
        win_rate = wins / max(trades, 1)
        
        # Calculate proper drawdown
        if len(balance_history) > 1:
            balance_series = pd.Series(balance_history)
            running_max = balance_series.expanding().max()
            drawdown_series = (running_max - balance_series) / running_max
            max_drawdown = drawdown_series.max()
            
            # Calculate Sharpe ratio
            balance_returns = balance_series.pct_change().dropna()
            sharpe = balance_returns.mean() / balance_returns.std() * np.sqrt(252) if balance_returns.std() > 0 else 0
        else:
            sharpe = 0
        
        # Analyze strategy performance
        strategy_stats = {}
        for strat_type, returns in strategy_performance.items():
            if returns:
                strategy_stats[strat_type] = {
                    'count': len(returns),
                    'avg_return': np.mean(returns),
                    'win_rate': len([r for r in returns if r > 0]) / len(returns),
                    'best_return': max(returns),
                    'worst_return': min(returns)
                }
        
        # Regime analysis
        regime_df = pd.DataFrame(regime_log)
        bull_periods = len(regime_df[regime_df['trend'].str.contains('bull')])
        bear_periods = len(regime_df[regime_df['trend'].str.contains('bear')])
        sideways_periods = len(regime_df[regime_df['trend'] == 'sideways'])
        avg_confidence = regime_df['confidence'].mean()
        
        print(f"✅ Trend-regime strategy complete!")
        print(f"   Final balance: ${balance:,.2f}")
        print(f"   Total return: {total_return:.1%}")
        print(f"   Trades: {trades}, Win rate: {win_rate:.1%}")
        print(f"   Max drawdown: {max_drawdown:.1%}")
        print(f"   Sharpe ratio: {sharpe:.2f}")
        print(f"   Regime periods: {bull_periods} bull, {bear_periods} bear, {sideways_periods} sideways")
        
        # Print strategy breakdown
        print(f"\n📊 STRATEGY PERFORMANCE BREAKDOWN:")
        for strat_type, stats in strategy_stats.items():
            print(f"   {strat_type}: {stats['count']} trades, {stats['avg_return']:+.1%} avg, {stats['win_rate']:.1%} win rate")
        
        # Enhanced fitness calculation
        if total_return > 0:
            # Reward higher returns with good risk management
            return_component = total_return * 0.6
            risk_component = (1 - max_drawdown) * 0.3
            consistency_component = win_rate * 0.1
            
            fitness = return_component + risk_component + consistency_component
            fitness = max(0.01, min(1.0, fitness))
        else:
            fitness = 0.01
        
        return {
            'fitness': fitness,
            'returns': total_return,
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'final_balance': balance,
            'trade_log': trade_log,
            'regime_log': regime_log,
            'balance_history': balance_history,
            'strategy_stats': strategy_stats,
            'regime_stats': {
                'bull_periods': bull_periods,
                'bear_periods': bear_periods,
                'sideways_periods': sideways_periods,
                'avg_confidence': avg_confidence,
                'total_periods': len(regime_log)
            }
        }
        
    except Exception as e:
        print(f"❌ Trend-regime strategy error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'fitness': 0.01,
            'returns': -1.0,
            'trades': 0,
            'error': str(e)
        }

def test_trend_regime_bitcoin():
    """Test the trend-following + regime-aware Bitcoin strategy"""
    print("₿ TESTING TREND-FOLLOWING + REGIME-AWARE BITCOIN STRATEGY")
    print("=" * 70)
    
    # Fetch Bitcoin data
    API_KEY = '4c322523f98c6c20dbe789194197dafac7329ec5a7dc378503118d443e867c2b'
    
    import requests
    end_date = pd.Timestamp.today() - pd.Timedelta(days=1)
    url = (
        f"https://min-api.cryptocompare.com/data/v2/histoday?"
        f"fsym=BTC&tsym=USD&limit=365&toTs={int(end_date.timestamp())}&api_key={API_KEY}"
    )
    
    try:
        print("📊 Fetching Bitcoin data...")
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()['Data']['Data']
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        df = df.rename(columns={
            'close': 'close', 'high': 'high', 'low': 'low', 'open': 'open'
        })
        
        numeric_columns = ['close', 'high', 'low', 'open']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        print(f"✅ Got {len(df)} days of Bitcoin data")
        print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
        
        # Test trend + regime strategy
        print(f"\n🚀 Testing trend-following + regime-aware strategy...")
        
        params = BitcoinTrendRegimeParams(
            # Trend-following (bull markets)
            trend_breakout_pct=0.60,           # Buy breakouts above 60% of range
            trend_stop_loss_pct=0.10,          # 10% stop loss for trends
            trend_position_size=1.4,           # Larger positions in trends
            
            # Mean-reversion (bear/sideways)
            reversion_buy_pct=0.30,            # Buy dips at 30% of range
            reversion_sell_pct=0.70,           # Sell at 70% of range
            reversion_stop_loss_pct=0.05,      # 5% stop for mean reversion
            reversion_position_size=0.8,       # Smaller positions
            
            # Risk management
            max_position_size=1.5,             # Max 1.5x leverage
            max_trades_per_regime=8,           # Limit overtrading
            
            # Regime thresholds
            bull_confidence_threshold=0.4,     # 40% confidence for bull trades
            bear_confidence_threshold=0.5      # 50% confidence for bear trades
        )
        
        results = evaluate_bitcoin_trend_regime_strategy(df, params)
        
        print(f"\n📊 TREND + REGIME RESULTS:")
        print(f"   Return: {results['returns']:.1%}")
        print(f"   Trades: {results['trades']}")
        print(f"   Win rate: {results['win_rate']:.1%}")
        print(f"   Max drawdown: {results['max_drawdown']:.1%}")
        print(f"   Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Fitness: {results['fitness']:.4f}")
        
        # Compare to market return
        market_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
        outperformance = results['returns'] - market_return
        
        print(f"\n📈 MARKET COMPARISON:")
        print(f"   Market return: {market_return:.1%}")
        print(f"   Strategy return: {results['returns']:.1%}")
        print(f"   Outperformance: {outperformance:+.1%}")
        
        # Performance verdict
        if outperformance > 0:
            verdict = "🎉 BEATS MARKET!"
        elif results['returns'] > market_return * 0.7:  # Within 30% of market
            verdict = "🔥 STRONG PERFORMANCE!"
        elif results['returns'] > 0:
            verdict = "✅ PROFITABLE (but underperforms market)"
        else:
            verdict = "❌ UNPROFITABLE"
        
        print(f"   Verdict: {verdict}")
        
        # Compare to previous strategies
        print(f"\n🔄 STRATEGY EVOLUTION:")
        print(f"   Original Bitcoin (overfitted): 24.6%")
        print(f"   Validated Bitcoin (realistic): 1.9%")
        print(f"   Regime-aware (mean-reversion): 1.9%")
        print(f"   Trend + Regime (adaptive): {results['returns']:.1%}")
        print(f"   Market benchmark: {market_return:.1%}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_trend_regime_bitcoin()
