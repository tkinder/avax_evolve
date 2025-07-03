#!/usr/bin/env python3
"""
Bitcoin Regime-Aware Backtester - FIXED VERSION
Integrates AVAX's sophisticated regime detection into Bitcoin strategy
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
class BitcoinRegimeParams:
    """Bitcoin strategy parameters with regime awareness"""
    # Core strategy parameters
    risk_reward: float = 1.5
    trend_strength: float = 1.2
    entry_threshold: float = 0.8
    confidence: float = 1.5
    
    # Buy/sell thresholds (base levels)
    buy_threshold_pct: float = 0.35
    sell_threshold_pct: float = 0.75
    
    # Risk management
    max_position_pct: float = 1.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.08
    
    # Regime multipliers (NEW!)
    bull_multiplier: float = 1.2      # More aggressive in bull markets
    bear_multiplier: float = 0.7      # More conservative in bear markets
    high_vol_multiplier: float = 0.6  # Reduce exposure in high volatility
    low_vol_multiplier: float = 1.3   # Increase exposure in low volatility
    
    # Regime windows
    regime_short_window: int = 10
    regime_medium_window: int = 30
    regime_long_window: int = 60
    
    # Regime-specific parameters
    regime_confidence_threshold: float = 0.3  # Minimum confidence to trade
    regime_avoid_extreme_vol: bool = True     # Avoid trading in extreme volatility
    regime_bear_market_filter: bool = True    # Reduce trading in bear markets

def evaluate_bitcoin_regime_strategy_fixed(df: pd.DataFrame, 
                                         params: BitcoinRegimeParams, 
                                         fitness_config=None) -> dict:
    """
    FIXED: Evaluate Bitcoin strategy with sophisticated regime awareness
    """
    try:
        print(f"🧠 Running FIXED regime-aware Bitcoin backtest...")
        
        # Initialize regime detector
        regime_detector = EnhancedRegimeDetector(
            short_window=params.regime_short_window,
            medium_window=params.regime_medium_window, 
            long_window=params.regime_long_window
        )
        
        # Setup
        initial_balance = 10000.0
        balance = initial_balance
        position = 0.0  # 0 = no position, >0 = long position size
        entry_price = 0.0
        entry_regime = None
        trades, wins, losses = 0, 0, 0
        peak_balance = initial_balance
        max_drawdown = 0.0
        
        # Calculate dynamic price levels
        price_min = df['close'].min()
        price_max = df['close'].max()
        price_range = price_max - price_min
        
        trade_log = []
        regime_log = []
        balance_history = [initial_balance]
        
        print(f"   Price range: ${price_min:,.0f} - ${price_max:,.0f}")
        print(f"   Starting balance: ${initial_balance:,.2f}")
        
        for i in range(params.regime_long_window, len(df)):
            current_price = df['close'].iloc[i]
            current_date = df.index[i]
            
            # Detect current market regime
            regime = regime_detector.detect_regime(df, i)
            regime_log.append({
                'date': current_date,
                'regime': str(regime),
                'confidence': regime.confidence,
                'price': current_price
            })
            
            # Calculate regime-adjusted thresholds
            base_buy_level = price_min + price_range * params.buy_threshold_pct
            base_sell_level = price_min + price_range * params.sell_threshold_pct
            
            # Apply regime multipliers
            position_multiplier = RegimeBasedTradingRules.get_position_size_multiplier(regime)
            
            # Regime-adjusted buy/sell levels
            if regime.is_bullish():
                buy_multiplier = params.bull_multiplier
                sell_multiplier = 1.0 / params.bull_multiplier  # Easier to sell in bull market
            elif regime.is_bearish():
                buy_multiplier = params.bear_multiplier
                sell_multiplier = 1.0 / params.bear_multiplier  # Harder to sell in bear market
            else:
                buy_multiplier = 1.0
                sell_multiplier = 1.0
            
            if regime.is_high_volatility():
                vol_multiplier = params.high_vol_multiplier
            else:
                vol_multiplier = params.low_vol_multiplier
            
            # Combined multiplier
            combined_buy_multiplier = buy_multiplier * vol_multiplier
            combined_sell_multiplier = sell_multiplier / vol_multiplier
            
            # Adjust thresholds based on regime
            adj_buy_level = base_buy_level * combined_buy_multiplier
            adj_sell_level = base_sell_level * combined_sell_multiplier
            
            # ENTRY LOGIC with regime awareness
            if position == 0:
                should_enter = RegimeBasedTradingRules.should_enter_long(
                    regime, current_price, adj_buy_level
                )
                
                # Additional regime filters
                if params.regime_avoid_extreme_vol and regime.volatility == VolatilityRegime.EXTREME:
                    should_enter = False
                
                if params.regime_bear_market_filter and regime.trend == TrendRegime.STRONG_BEAR:
                    should_enter = False
                
                if regime.confidence < params.regime_confidence_threshold:
                    should_enter = False
                
                if should_enter:
                    # Execute entry
                    position_size = min(params.max_position_pct * position_multiplier, 1.5)
                    entry_price = current_price * 1.0005  # Small slippage
                    position = position_size
                    entry_regime = regime
                    trades += 1
                    
                    trade_log.append({
                        'type': 'BUY',
                        'date': current_date,
                        'price': entry_price,
                        'position_size': position_size,
                        'regime': str(regime),
                        'confidence': regime.confidence,
                        'adj_buy_level': adj_buy_level,
                        'multiplier': combined_buy_multiplier,
                        'balance_before': balance
                    })
                    
                    if len(trade_log) <= 5:  # Only print first 5 trades
                        print(f"   BUY: ${entry_price:,.0f} on {current_date.strftime('%Y-%m-%d')} "
                              f"(regime: {regime.trend.value}, pos: {position_size:.1f}x, bal: ${balance:,.0f})")
            
            # EXIT LOGIC with regime awareness
            else:
                should_exit, exit_reason = RegimeBasedTradingRules.should_exit_long(
                    regime, current_price, entry_price, 
                    adj_sell_level, params.stop_loss_pct, params.take_profit_pct
                )
                
                if should_exit:
                    # Execute exit - FIXED CALCULATION
                    exit_price = current_price * 0.9995  # Small slippage
                    gross_return = (exit_price - entry_price) / entry_price
                    net_return = gross_return - 0.002  # Fees (0.1% each side)
                    
                    # CORRECT balance calculation
                    position_return = net_return * position
                    balance *= (1 + position_return)  # Compound the return
                    
                    is_win = net_return > 0
                    wins += 1 if is_win else 0
                    losses += 1 if not is_win else 0
                    
                    trade_log.append({
                        'type': 'SELL',
                        'date': current_date,
                        'price': exit_price,
                        'gross_return': gross_return,
                        'net_return': net_return,
                        'position_return': position_return,
                        'reason': exit_reason,
                        'regime': str(regime),
                        'entry_regime': str(entry_regime) if entry_regime else 'unknown',
                        'balance_before': balance / (1 + position_return),  # Before this trade
                        'balance_after': balance
                    })
                    
                    if len([t for t in trade_log if t['type'] == 'SELL']) <= 5:  # Only print first 5 exits
                        print(f"   SELL: ${exit_price:,.0f} on {current_date.strftime('%Y-%m-%d')} "
                              f"({net_return:+.1%}, reason: {exit_reason}, bal: ${balance:,.0f})")
                    
                    position = 0
                    entry_price = 0
                    entry_regime = None
            
            # Track balance and drawdown
            balance_history.append(balance)
            peak_balance = max(peak_balance, balance)
            current_drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, current_drawdown)
        
        # Force exit if still in position
        if position > 0:
            final_price = df['close'].iloc[-1] * 0.9995
            gross_return = (final_price - entry_price) / entry_price
            net_return = gross_return - 0.002
            position_return = net_return * position
            balance *= (1 + position_return)
            
            is_win = net_return > 0
            wins += 1 if is_win else 0
            losses += 1 if not is_win else 0
            
            print(f"   FORCED EXIT: ${final_price:,.0f} ({net_return:+.1%}, bal: ${balance:,.0f})")
        
        # Calculate metrics
        total_return = (balance - initial_balance) / initial_balance
        win_rate = wins / max(trades, 1)
        
        # Calculate proper drawdown from balance history
        balance_series = pd.Series(balance_history)
        running_max = balance_series.expanding().max()
        drawdown_series = (running_max - balance_series) / running_max
        max_drawdown = drawdown_series.max()
        
        # Regime analysis
        regime_df = pd.DataFrame(regime_log)
        bull_periods = len(regime_df[regime_df['regime'].str.contains('bull')])
        bear_periods = len(regime_df[regime_df['regime'].str.contains('bear')])
        avg_confidence = regime_df['confidence'].mean()
        
        # Calculate Sharpe ratio
        if len(balance_history) > 1:
            balance_returns = pd.Series(balance_history).pct_change().dropna()
            sharpe = balance_returns.mean() / balance_returns.std() * np.sqrt(252) if balance_returns.std() > 0 else 0
        else:
            sharpe = 0
        
        print(f"✅ FIXED regime-aware backtest complete!")
        print(f"   Final balance: ${balance:,.2f}")
        print(f"   Total return: {total_return:.1%}")
        print(f"   Trades: {trades}, Win rate: {win_rate:.1%}")
        print(f"   Max drawdown: {max_drawdown:.1%}")
        print(f"   Sharpe ratio: {sharpe:.2f}")
        print(f"   Regime periods: {bull_periods} bull, {bear_periods} bear")
        print(f"   Average regime confidence: {avg_confidence:.2f}")
        
        # Enhanced fitness calculation
        if total_return > 0 and max_drawdown < 0.5:
            base_fitness = total_return * win_rate * (1 - max_drawdown)
            regime_bonus = avg_confidence * 0.1  # Bonus for high-confidence trading
            trade_frequency_penalty = max(0, (trades - 50) * 0.001)  # Penalty for overtrading
            fitness = max(0.01, base_fitness + regime_bonus - trade_frequency_penalty)
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
            'regime_stats': {
                'bull_periods': bull_periods,
                'bear_periods': bear_periods,
                'avg_confidence': avg_confidence,
                'total_periods': len(regime_log)
            }
        }
        
    except Exception as e:
        print(f"❌ FIXED regime backtest error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'fitness': 0.01,
            'returns': -1.0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'max_drawdown': 1.0,
            'sharpe_ratio': 0.0,
            'final_balance': 0.0,
            'error': str(e)
        }

def test_fixed_regime_aware_bitcoin():
    """Test the FIXED regime-aware Bitcoin strategy"""
    print("₿ TESTING FIXED REGIME-AWARE BITCOIN STRATEGY")
    print("=" * 60)
    
    # Fetch Bitcoin data using CryptoCompare
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
        
        # Test FIXED regime-aware strategy
        print(f"\n🧠 Testing FIXED regime-aware strategy...")
        
        params = BitcoinRegimeParams(
            buy_threshold_pct=0.30,        # Buy when price is in bottom 30% of range
            sell_threshold_pct=0.70,       # Sell when price is in top 70% of range
            bull_multiplier=1.3,           # More aggressive in bull markets
            bear_multiplier=0.6,           # More conservative in bear markets
            high_vol_multiplier=0.5,       # Reduce exposure in high volatility
            low_vol_multiplier=1.4,        # Increase exposure in low volatility
            stop_loss_pct=0.03,            # 3% stop loss
            take_profit_pct=0.10,          # 10% take profit
            regime_confidence_threshold=0.4 # Require 40% confidence to trade
        )
        
        results = evaluate_bitcoin_regime_strategy_fixed(df, params)
        
        print(f"\n📊 FIXED REGIME-AWARE RESULTS:")
        print(f"   Return: {results['returns']:.1%}")
        print(f"   Trades: {results['trades']}")
        print(f"   Win rate: {results['win_rate']:.1%}")
        print(f"   Max drawdown: {results['max_drawdown']:.1%}")
        print(f"   Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Fitness: {results['fitness']:.4f}")
        
        if 'regime_stats' in results:
            stats = results['regime_stats']
            print(f"   Bull periods: {stats['bull_periods']}")
            print(f"   Bear periods: {stats['bear_periods']}")
            print(f"   Avg confidence: {stats['avg_confidence']:.2f}")
        
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
        elif results['returns'] > 0:
            verdict = "✅ PROFITABLE (but underperforms market)"
        else:
            verdict = "❌ UNPROFITABLE"
        
        print(f"   Verdict: {verdict}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_fixed_regime_aware_bitcoin()
