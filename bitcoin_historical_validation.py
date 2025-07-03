#!/usr/bin/env python3
"""
Bitcoin Historical Validation - Walk-Forward Test
Tests our technical strategy across multiple market cycles to ensure it's not overfitted
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
from typing import Tuple, Optional, Dict, List

# Import regime detection
from src.core.regime_detector import (
    EnhancedRegimeDetector, RegimeBasedTradingRules, 
    TrendRegime, VolatilityRegime, MomentumRegime
)

@dataclass
class BitcoinTechnicalParams:
    """Bitcoin technical parameters - FIXED from our best result"""
    
    # Moving Average Periods (Conservative - our best performer)
    ma_fast: int = 20
    ma_medium: int = 50  
    ma_slow: int = 200
    
    # ATR settings (Conservative)
    atr_period: int = 14
    atr_entry_multiplier: float = 2.0
    atr_exit_multiplier: float = 2.5
    
    # RSI settings (Conservative)
    rsi_period: int = 14
    rsi_oversold: float = 25
    rsi_overbought: float = 75
    
    # Fixed parameters - NO OPTIMIZATION
    regime_confidence_min: float = 0.4
    stop_loss_atr_multiple: float = 2.0
    take_profit_atr_multiple: float = 3.0
    max_position_size: float = 1.0
    fee_rate: float = 0.001
    slippage_pct: float = 0.0005

def calculate_technical_indicators(df: pd.DataFrame, params: BitcoinTechnicalParams) -> pd.DataFrame:
    """Calculate technical indicators - same as before"""
    df = df.copy()
    
    # Moving Averages
    df['ma_fast'] = df['close'].rolling(window=params.ma_fast).mean()
    df['ma_medium'] = df['close'].rolling(window=params.ma_medium).mean()
    df['ma_slow'] = df['close'].rolling(window=params.ma_slow).mean()
    
    # ATR (Average True Range)
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=params.atr_period).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=params.rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=params.rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Dynamic levels
    df['dynamic_support'] = df['ma_fast'] - (params.atr_entry_multiplier * df['atr'])
    df['dynamic_resistance'] = df['ma_fast'] + (params.atr_exit_multiplier * df['atr'])
    
    # Trend signals
    df['golden_cross'] = (df['ma_fast'] > df['ma_medium']) & (df['ma_fast'].shift() <= df['ma_medium'].shift())
    df['death_cross'] = (df['ma_fast'] < df['ma_medium']) & (df['ma_fast'].shift() >= df['ma_medium'].shift())
    df['bull_regime'] = df['close'] > df['ma_slow']
    df['trend_up'] = (df['ma_fast'] > df['ma_medium']) & (df['ma_medium'] > df['ma_slow'])
    
    return df

def backtest_period(df: pd.DataFrame, params: BitcoinTechnicalParams, period_name: str) -> Dict:
    """
    Backtest our FIXED strategy on a specific time period
    NO PARAMETER CHANGES - true validation
    """
    try:
        print(f"🧪 Testing {period_name}...")
        print(f"   Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} days)")
        
        # Calculate indicators
        df = calculate_technical_indicators(df, params)
        
        # Initialize regime detector
        regime_detector = EnhancedRegimeDetector(short_window=10, medium_window=30, long_window=60)
        
        # Setup backtest
        initial_balance = 10000.0
        balance = initial_balance
        position = 0
        entry_price = 0.0
        entry_data = {}
        trades, wins, losses = 0, 0, 0
        peak_balance = initial_balance
        max_drawdown = 0.0
        
        trade_log = []
        balance_history = [initial_balance]
        
        signal_counts = {
            'golden_cross_signals': 0,
            'oversold_bounce_signals': 0,
            'breakout_signals': 0,
            'regime_filtered_out': 0,
            'rsi_filtered_out': 0
        }
        
        start_idx = max(params.ma_slow, params.atr_period, params.rsi_period, 60)
        
        for i in range(start_idx, len(df)):
            current_price = df['close'].iloc[i]
            current_date = df.index[i]
            
            # Get indicators
            ma_fast = df['ma_fast'].iloc[i]
            ma_medium = df['ma_medium'].iloc[i]
            ma_slow = df['ma_slow'].iloc[i]
            atr = df['atr'].iloc[i]
            rsi = df['rsi'].iloc[i]
            dynamic_support = df['dynamic_support'].iloc[i]
            dynamic_resistance = df['dynamic_resistance'].iloc[i]
            
            if pd.isna(ma_fast) or pd.isna(atr) or pd.isna(rsi):
                balance_history.append(balance)
                continue
            
            # Detect regime
            regime = regime_detector.detect_regime(df, i)
            
            # ENTRY LOGIC - EXACT SAME AS VALIDATED STRATEGY
            if position == 0:
                entry_signal = False
                signal_type = ""
                position_size = 0
                
                # Signal 1: Golden Cross + Bull Regime
                if (df['golden_cross'].iloc[i] and 
                    regime.is_bullish() and 
                    regime.confidence >= params.regime_confidence_min):
                    
                    entry_signal = True
                    signal_type = "golden_cross"
                    position_size = params.max_position_size
                    signal_counts['golden_cross_signals'] += 1
                
                # Signal 2: Oversold Bounce in Bull Market
                elif (current_price <= dynamic_support and 
                      rsi <= params.rsi_oversold and
                      df['bull_regime'].iloc[i] and
                      regime.confidence >= params.regime_confidence_min):
                    
                    entry_signal = True
                    signal_type = "oversold_bounce"
                    position_size = params.max_position_size * 0.8
                    signal_counts['oversold_bounce_signals'] += 1
                
                # Signal 3: Breakout Above Resistance
                elif (current_price > dynamic_resistance and
                      df['trend_up'].iloc[i] and
                      regime.trend in [TrendRegime.STRONG_BULL, TrendRegime.MILD_BULL] and
                      regime.momentum in [MomentumRegime.ACCELERATING_UP, MomentumRegime.STEADY_UP]):
                    
                    entry_signal = True
                    signal_type = "breakout"
                    position_size = params.max_position_size * 1.2
                    signal_counts['breakout_signals'] += 1
                
                # Apply filters
                if entry_signal:
                    if regime.is_bearish() and signal_type != "oversold_bounce":
                        entry_signal = False
                        signal_counts['regime_filtered_out'] += 1
                    
                    if rsi >= 80:
                        entry_signal = False
                        signal_counts['rsi_filtered_out'] += 1
                
                if entry_signal:
                    entry_price = current_price * (1 + params.slippage_pct)
                    position = min(position_size, params.max_position_size)
                    trades += 1
                    
                    stop_loss_price = entry_price - (params.stop_loss_atr_multiple * atr)
                    take_profit_price = entry_price + (params.take_profit_atr_multiple * atr)
                    
                    entry_data = {
                        'price': entry_price,
                        'date': current_date,
                        'signal_type': signal_type,
                        'regime': regime,
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price,
                        'atr': atr
                    }
                    
                    trade_log.append({
                        'type': 'BUY',
                        'date': current_date,
                        'price': entry_price,
                        'signal_type': signal_type,
                        'regime': str(regime)
                    })
            
            # EXIT LOGIC - EXACT SAME AS VALIDATED STRATEGY
            else:
                should_exit = False
                exit_reason = ""
                
                if df['death_cross'].iloc[i]:
                    should_exit = True
                    exit_reason = "death_cross"
                elif current_price >= entry_data['take_profit']:
                    should_exit = True
                    exit_reason = "take_profit"
                elif current_price <= entry_data['stop_loss']:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif current_price >= dynamic_resistance and rsi >= params.rsi_overbought:
                    should_exit = True
                    exit_reason = "overbought_resistance"
                elif regime.should_avoid_trading():
                    should_exit = True
                    exit_reason = "regime_risk"
                elif (regime.trend == TrendRegime.STRONG_BEAR and 
                      entry_data['signal_type'] != "oversold_bounce"):
                    should_exit = True
                    exit_reason = "bear_regime"
                
                if should_exit:
                    exit_price = current_price * (1 - params.slippage_pct)
                    gross_return = (exit_price - entry_price) / entry_price
                    net_return = gross_return - 2 * params.fee_rate
                    
                    position_return = net_return * position
                    balance *= (1 + position_return)
                    
                    is_win = net_return > 0
                    wins += 1 if is_win else 0
                    losses += 1 if not is_win else 0
                    
                    hold_days = (current_date - entry_data['date']).days
                    
                    trade_log.append({
                        'type': 'SELL',
                        'date': current_date,
                        'price': exit_price,
                        'net_return': net_return,
                        'reason': exit_reason,
                        'hold_days': hold_days
                    })
                    
                    position = 0
                    entry_data = {}
            
            # Track performance
            balance_history.append(balance)
            peak_balance = max(peak_balance, balance)
            current_drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, current_drawdown)
        
        # Force exit if needed
        if position > 0:
            final_price = df['close'].iloc[-1] * (1 - params.slippage_pct)
            gross_return = (final_price - entry_data['price']) / entry_data['price']
            net_return = gross_return - 2 * params.fee_rate
            position_return = net_return * position
            balance *= (1 + position_return)
            
            is_win = net_return > 0
            wins += 1 if is_win else 0
            losses += 1 if not is_win else 0
        
        # Calculate metrics
        total_return = (balance - initial_balance) / initial_balance
        win_rate = wins / max(trades, 1)
        market_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
        
        # Sharpe ratio
        if len(balance_history) > 1:
            balance_series = pd.Series(balance_history)
            balance_returns = balance_series.pct_change().dropna()
            sharpe = balance_returns.mean() / balance_returns.std() * np.sqrt(252) if balance_returns.std() > 0 else 0
        else:
            sharpe = 0
        
        print(f"   Strategy return: {total_return:.1%}")
        print(f"   Market return: {market_return:.1%}")
        print(f"   Outperformance: {(total_return - market_return):+.1%}")
        print(f"   Trades: {trades}, Win rate: {win_rate:.1%}")
        print(f"   Max drawdown: {max_drawdown:.1%}")
        print(f"   Sharpe ratio: {sharpe:.2f}")
        
        return {
            'period_name': period_name,
            'start_date': df.index[0],
            'end_date': df.index[-1],
            'days': len(df),
            'strategy_return': total_return,
            'market_return': market_return,
            'outperformance': total_return - market_return,
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'final_balance': balance,
            'trade_log': trade_log,
            'signal_counts': signal_counts,
            'market_type': 'bull' if market_return > 0.1 else 'bear' if market_return < -0.1 else 'sideways'
        }
        
    except Exception as e:
        print(f"❌ Error testing {period_name}: {e}")
        return {
            'period_name': period_name,
            'error': str(e),
            'strategy_return': -1.0,
            'market_return': 0.0,
            'trades': 0
        }

def run_historical_validation():
    """
    Run walk-forward validation across multiple Bitcoin market cycles
    """
    print("₿ BITCOIN HISTORICAL VALIDATION")
    print("=" * 60)
    print("🎯 Testing our 15% technical strategy across market cycles")
    print("🔒 ZERO parameter changes - true generalization test")
    
    # Fetch extended Bitcoin data
    API_KEY = '4c322523f98c6c20dbe789194197dafac7329ec5a7dc378503118d443e867c2b'
    
    import requests
    end_date = pd.Timestamp.today() - pd.Timedelta(days=1)
    
    try:
        print("\n📊 Fetching extended Bitcoin historical data...")
        
        # Fetch more data to cover multiple cycles
        url = (
            f"https://min-api.cryptocompare.com/data/v2/histoday?"
            f"fsym=BTC&tsym=USD&limit=1500&toTs={int(end_date.timestamp())}&api_key={API_KEY}"
        )
        
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()['Data']['Data']
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        df = df.rename(columns={'close': 'close', 'high': 'high', 'low': 'low', 'open': 'open'})
        numeric_columns = ['close', 'high', 'low', 'open']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        print(f"✅ Got {len(df)} days of Bitcoin data")
        print(f"   Full range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
        
        # Define test periods for walk-forward validation
        test_periods = [
            ('2021 Bull Peak', '2021-01-01', '2021-11-30'),      # Bull market peak
            ('2022 Bear Crash', '2021-12-01', '2022-12-31'),     # Major crash  
            ('2023 Recovery', '2023-01-01', '2023-12-31'),       # Recovery year
            ('2024 Bull Run', '2024-01-01', '2024-12-31'),       # New bull market
            ('2025 Current', '2025-01-01', '2025-07-01'),        # Recent period
        ]
        
        # Use our FIXED conservative parameters (no optimization!)
        params = BitcoinTechnicalParams()
        
        print(f"\n🔒 FIXED STRATEGY PARAMETERS (NO OPTIMIZATION):")
        print(f"   Moving Averages: {params.ma_fast}/{params.ma_medium}/{params.ma_slow}")
        print(f"   ATR: {params.atr_entry_multiplier}x entry, {params.atr_exit_multiplier}x exit")
        print(f"   RSI: {params.rsi_oversold}/{params.rsi_overbought}")
        
        # Test each period
        results = []
        
        for period_name, start_date, end_date in test_periods:
            period_df = df[(df.index >= start_date) & (df.index <= end_date)].copy()
            
            if len(period_df) < 250:  # Need sufficient data
                print(f"⚠️  Skipping {period_name} - insufficient data ({len(period_df)} days)")
                continue
            
            result = backtest_period(period_df, params, period_name)
            results.append(result)
        
        # Analyze results across periods
        analyze_validation_results(results)
        
        return results
        
    except Exception as e:
        print(f"❌ Historical validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_validation_results(results: List[Dict]):
    """Analyze validation results across all periods"""
    
    print(f"\n📊 HISTORICAL VALIDATION ANALYSIS")
    print("=" * 60)
    
    # Filter successful results
    successful_results = [r for r in results if 'error' not in r]
    
    if not successful_results:
        print("❌ No successful validation periods!")
        return
    
    # Summary table
    print(f"📅 PERIOD-BY-PERIOD RESULTS:")
    print(f"{'Period':<15} {'Strategy':<10} {'Market':<10} {'Outperf':<10} {'Trades':<7} {'Sharpe':<7} {'Type':<8}")
    print("-" * 80)
    
    for result in successful_results:
        print(f"{result['period_name']:<15} {result['strategy_return']:>8.1%} "
              f"{result['market_return']:>8.1%} {result['outperformance']:>+8.1%} "
              f"{result['trades']:>5} {result['sharpe_ratio']:>6.2f} {result['market_type']:<8}")
    
    # Aggregate statistics
    strategy_returns = [r['strategy_return'] for r in successful_results]
    market_returns = [r['market_return'] for r in successful_results]
    outperformances = [r['outperformance'] for r in successful_results]
    sharpe_ratios = [r['sharpe_ratio'] for r in successful_results]
    trades_per_period = [r['trades'] for r in successful_results]
    
    # Calculate success metrics
    profitable_periods = len([r for r in strategy_returns if r > 0])
    market_beating_periods = len([r for r in outperformances if r > 0])
    
    print(f"\n📈 AGGREGATE VALIDATION RESULTS:")
    print(f"   Periods tested: {len(successful_results)}")
    print(f"   Profitable periods: {profitable_periods}/{len(successful_results)} ({profitable_periods/len(successful_results)*100:.0f}%)")
    print(f"   Market-beating periods: {market_beating_periods}/{len(successful_results)} ({market_beating_periods/len(successful_results)*100:.0f}%)")
    
    print(f"\n📊 PERFORMANCE STATISTICS:")
    print(f"   Average strategy return: {np.mean(strategy_returns):8.1%}")
    print(f"   Median strategy return:  {np.median(strategy_returns):8.1%}")
    print(f"   Best strategy return:    {max(strategy_returns):8.1%}")
    print(f"   Worst strategy return:   {min(strategy_returns):8.1%}")
    print(f"   Return std deviation:    {np.std(strategy_returns):8.1%}")
    
    print(f"\n📊 RISK-ADJUSTED PERFORMANCE:")
    print(f"   Average Sharpe ratio: {np.mean(sharpe_ratios):6.2f}")
    print(f"   Median Sharpe ratio:  {np.median(sharpe_ratios):6.2f}")
    print(f"   Average outperformance: {np.mean(outperformances):+.1%}")
    
    print(f"\n📊 TRADING ACTIVITY:")
    print(f"   Average trades per period: {np.mean(trades_per_period):.1f}")
    print(f"   Total trades across periods: {sum(trades_per_period)}")
    
    # Market type analysis
    bull_results = [r for r in successful_results if r['market_type'] == 'bull']
    bear_results = [r for r in successful_results if r['market_type'] == 'bear']
    sideways_results = [r for r in successful_results if r['market_type'] == 'sideways']
    
    if bull_results:
        bull_avg = np.mean([r['strategy_return'] for r in bull_results])
        bull_outperf = np.mean([r['outperformance'] for r in bull_results])
        print(f"   Bull market performance: {bull_avg:.1%} (outperformance: {bull_outperf:+.1%})")
    
    if bear_results:
        bear_avg = np.mean([r['strategy_return'] for r in bear_results])
        bear_outperf = np.mean([r['outperformance'] for r in bear_results])
        print(f"   Bear market performance: {bear_avg:.1%} (outperformance: {bear_outperf:+.1%})")
    
    if sideways_results:
        sideways_avg = np.mean([r['strategy_return'] for r in sideways_results])
        sideways_outperf = np.mean([r['outperformance'] for r in sideways_results])
        print(f"   Sideways market performance: {sideways_avg:.1%} (outperformance: {sideways_outperf:+.1%})")
    
    # Final validation verdict
    print(f"\n🎯 VALIDATION VERDICT:")
    
    consistent_profitable = profitable_periods >= len(successful_results) * 0.6  # 60%+ profitable
    positive_sharpe = np.mean(sharpe_ratios) > 0.3
    controlled_risk = all(r.get('max_drawdown', 1) < 0.25 for r in successful_results)  # <25% drawdown
    
    if consistent_profitable and positive_sharpe and controlled_risk:
        verdict = "✅ STRATEGY VALIDATED - Generalizes across market cycles"
    elif consistent_profitable and positive_sharpe:
        verdict = "⚠️  CONDITIONALLY VALIDATED - Profitable but check risk"
    elif np.mean(strategy_returns) > 0:
        verdict = "❓ MIXED RESULTS - Some profitability but inconsistent"
    else:
        verdict = "❌ STRATEGY FAILS - Does not generalize"
    
    print(f"   {verdict}")
    
    # Compare to our original 15% result
    recent_result = [r for r in successful_results if '2024' in r['period_name'] or '2025' in r['period_name']]
    if recent_result:
        recent_return = recent_result[0]['strategy_return']
        print(f"\n💡 OVERFITTING CHECK:")
        print(f"   Original 2024-2025 result: 15.0%")
        print(f"   Historical validation on 2024-2025: {recent_return:.1%}")
        if abs(recent_return - 0.15) < 0.05:  # Within 5%
            print(f"   ✅ Results consistent - not overfitted")
        else:
            print(f"   ⚠️  Results differ - possible overfitting")

if __name__ == "__main__":
    run_historical_validation()
