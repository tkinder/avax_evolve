#!/usr/bin/env python3
"""
Bitcoin Regime-Aware Backtester
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

def evaluate_bitcoin_regime_strategy(df: pd.DataFrame, 
                                   params: BitcoinRegimeParams, 
                                   fitness_config=None) -> dict:
    """
    Evaluate Bitcoin strategy with sophisticated regime awareness
    """
    try:
        print(f"🧠 Running regime-aware Bitcoin backtest...")
        
        # Initialize regime detector
        regime_detector = EnhancedRegimeDetector(
            short_window=params.regime_short_window,
            medium_window=params.regime_medium_window, 
            long_window=params.regime_long_window
        )
        
        # Setup
        initial_balance = 10000
        balance = initial_balance
        position = 0
        entry_price = 0.0
        entry_regime = None
        trades, wins, losses = 0, 0, 0
        peak_balance = initial_balance
        max_drawdown = 0
        
        # Calculate dynamic price levels
        price_min = df['close'].min()
        price_max = df['close'].max()
        price_range = price_max - price_min
        
        trade_log = []
        regime_log = []
        
        print(f"   Price range: ${price_min:,.0f} - ${price_max:,.0f}")
        
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
                sell_multiplier = params.bull_multiplier
            elif regime.is_bearish():
                buy_multiplier = params.bear_multiplier
                sell_multiplier = params.bear_multiplier
            else:
                buy_multiplier = 1.0
                sell_multiplier = 1.0
            
            if regime.is_high_volatility():
                vol_multiplier = params.high_vol_multiplier
            else:
                vol_multiplier = params.low_vol_multiplier
            
            # Combined multiplier
            combined_multiplier = buy_multiplier * vol_multiplier
            
            # Adjust thresholds based on regime
            adj_buy_level = base_buy_level * combined_multiplier
            adj_sell_level = base_sell_level / combined_multiplier  # Easier to sell in good conditions
            
            prev_balance = balance
            
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
                    position_size = min(params.max_position_pct * position_multiplier, 2.0)
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
                        'multiplier': combined_multiplier
                    })
                    
                    print(f"   BUY: ${entry_price:,.0f} on {current_date.strftime('%Y-%m-%d')} "
                          f"(regime: {regime.trend.value}, pos: {position_size:.1f}x)")
            
            # EXIT LOGIC with regime awareness
            else:
                should_exit, exit_reason = RegimeBasedTradingRules.should_exit_long(
                    regime, current_price, entry_price, 
                    adj_sell_level, params.stop_loss_pct, params.take_profit_pct
                )
                
                if should_exit:
                    # Execute exit
                    exit_price = current_price * 0.9995  # Small slippage
                    gross_return = (exit_price - entry_price) / entry_price
                    net_return = gross_return - 0.002  # Fees
                    
                    balance = initial_balance + (balance - initial_balance) * (1 + net_return * position)
                    
                    is_win = net_return > 0
                    wins += 1 if is_win else 0
                    losses += 1 if not is_win else 0
                    
                    trade_log.append({
                        'type': 'SELL',
                        'date': current_date,
                        'price': exit_price,
                        'return': net_return,
                        'reason': exit_reason,
                        'regime': str(regime),
                        'entry_regime': str(entry_regime) if entry_regime else 'unknown'
                    })
                    
                    print(f"   SELL: ${exit_price:,.0f} on {current_date.strftime('%Y-%m-%d')} "
                          f"({net_return:+.1%}, reason: {exit_reason})")
                    
                    position = 0
                    entry_price = 0
                    entry_regime = None
            
            # Update drawdown
            peak_balance = max(peak_balance, balance)
            current_drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, current_drawdown)
        
        # Force exit if still in position
        if position > 0:
            final_price = df['close'].iloc[-1] * 0.9995
            gross_return = (final_price - entry_price) / entry_price
            net_return = gross_return - 0.002
            balance = initial_balance + (balance - initial_balance) * (1 + net_return * position)
            
            is_win = net_return > 0
            wins += 1 if is_win else 0
            losses += 1 if not is_win else 0
            trades += 1
            
            print(f"   FORCED EXIT: ${final_price:,.0f} ({net_return:+.1%})")
        
        # Calculate metrics
        total_return = (balance - initial_balance) / initial_balance
        win_rate = wins / max(trades, 1)
        
        # Regime analysis
        regime_df = pd.DataFrame(regime_log)
        bull_periods = len(regime_df[regime_df['regime'].str.contains('bull')])
        bear_periods = len(regime_df[regime_df['regime'].str.contains('bear')])
        avg_confidence = regime_df['confidence'].mean()
        
        print(f"✅ Regime-aware backtest complete!")
        print(f"   Final balance: ${balance:,.2f}")
        print(f"   Total return: {total_return:.1%}")
        print(f"   Trades: {trades}, Win rate: {win_rate:.1%}")
        print(f"   Max drawdown: {max_drawdown:.1%}")
        print(f"   Regime periods: {bull_periods} bull, {bear_periods} bear")
        print(f"   Average regime confidence: {avg_confidence:.2f}")
        
        # Enhanced fitness calculation (placeholder - can integrate with AVAX's fitness system)
        base_fitness = total_return * win_rate * (1 - max_drawdown)
        regime_bonus = avg_confidence * 0.1  # Bonus for high-confidence trading
        fitness = max(0.01, base_fitness + regime_bonus)
        
        return {
            'fitness': fitness,
            'returns': total_return,
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'final_balance': balance,
            'trade_log': trade_log,
            'regime_log': regime_log,
            'regime_stats': {
                'bull_periods': bull_periods,
                'bear_periods': bear_periods,
                'avg_confidence': avg_confidence,
                'total_periods': len(regime_log)
            }
        }
        
    except Exception as e:
        print(f"❌ Regime backtest error: {e}")
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
            'final_balance': 0.0,
            'error': str(e)
        }

def test_regime_aware_bitcoin():
    """Test the regime-aware Bitcoin strategy"""
    print("₿ TESTING REGIME-AWARE BITCOIN STRATEGY")
    print("=" * 50)
    
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
        
        # Test regime-aware strategy
        print(f"\n🧠 Testing regime-aware strategy...")
        
        params = BitcoinRegimeParams(
            buy_threshold_pct=0.35,
            sell_threshold_pct=0.75,
            bull_multiplier=1.3,
            bear_multiplier=0.6,
            high_vol_multiplier=0.5,
            low_vol_multiplier=1.4
        )
        
        results = evaluate_bitcoin_regime_strategy(df, params)
        
        print(f"\n📊 REGIME-AWARE RESULTS:")
        print(f"   Return: {results['returns']:.1%}")
        print(f"   Trades: {results['trades']}")
        print(f"   Win rate: {results['win_rate']:.1%}")
        print(f"   Max drawdown: {results['max_drawdown']:.1%}")
        print(f"   Fitness: {results['fitness']:.4f}")
        
        if 'regime_stats' in results:
            stats = results['regime_stats']
            print(f"   Bull periods: {stats['bull_periods']}")
            print(f"   Bear periods: {stats['bear_periods']}")
            print(f"   Avg confidence: {stats['avg_confidence']:.2f}")
        
        # Compare to market return
        market_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
        print(f"   Market return: {market_return:.1%}")
        print(f"   Outperformance: {results['returns'] - market_return:.1%}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    test_regime_aware_bitcoin()
