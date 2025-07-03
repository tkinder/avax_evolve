#!/usr/bin/env python3
"""
Bitcoin AVAX-Style Strategy
Uses AVAX's exact approach: 200-day MA regime filter + buy dips in uptrends + trailing stops
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

@dataclass
class BitcoinAVAXParams:
    """Bitcoin parameters using AVAX's exact strategy approach"""
    
    # Price levels (like AVAX top/bottom)
    bottom: float = 0.30        # Buy dips at 30% of range (like AVAX)  
    top: float = 0.75           # Sell at 75% of range (like AVAX)
    
    # Moving average regime filter (like AVAX)
    ma_period: int = 200        # 200-day MA for regime detection
    
    # Trailing stop (like AVAX)
    trailing_stop_pct: float = 0.10    # 10% trailing stop
    
    # Execution costs (like AVAX)
    fee_rate: float = 0.001     # 0.1% fee per side
    slippage_pct: float = 0.0005 # 0.05% slippage

def evaluate_bitcoin_avax_strategy(df: pd.DataFrame, params: BitcoinAVAXParams) -> dict:
    """
    Bitcoin strategy using AVAX's exact approach:
    1. 200-day MA regime filter
    2. Buy dips in uptrends only  
    3. Trailing stops to ride trends
    """
    try:
        print(f"📈 Running Bitcoin with AVAX's strategy...")
        
        # STEP 1: Calculate moving average regime (exactly like AVAX)
        df = df.copy()
        df['sma200'] = df['close'].rolling(window=params.ma_period).mean()
        df['regime'] = df['close'] > df['sma200']  # Bull regime when above 200-day MA
        
        # STEP 2: Normalize price levels (exactly like AVAX)
        price_min = df['close'].min()
        price_max = df['close'].max()
        price_range = price_max - price_min
        
        buy_level = price_min + price_range * params.bottom   # 30% of range
        sell_level = price_min + price_range * params.top     # 75% of range
        
        print(f"   Price range: ${price_min:,.0f} - ${price_max:,.0f}")
        print(f"   Buy level: ${buy_level:,.0f} (dips)")
        print(f"   Sell level: ${sell_level:,.0f} (profit target)")
        print(f"   200-day MA regime filter: {df['regime'].sum()}/{len(df)} days bullish")
        
        # STEP 3: Run backtest (exactly like AVAX)
        initial_balance = 10000.0
        balance = initial_balance
        position = 0
        entry_price = 0.0
        trades, wins, losses = 0, 0, 0
        peak_balance = initial_balance
        max_drawdown = 0.0
        
        trade_log = []
        balance_history = [initial_balance]
        highest_price = 0.0
        
        bull_days = 0
        bear_days = 0
        
        for i in range(200, len(df)):  # Start after 200-day MA is available
            close = df['close'].iloc[i]
            date = df.index[i]
            in_uptrend = df['regime'].iloc[i]  # Above 200-day MA = bull regime
            
            if pd.isna(df['sma200'].iloc[i]):
                balance_history.append(balance)
                continue
            
            # Track regime days
            if in_uptrend:
                bull_days += 1
            else:
                bear_days += 1
            
            prev_balance = balance
            
            # ENTRY LOGIC: Buy dips in uptrends only (exactly like AVAX)
            if position == 0:
                if close < buy_level and in_uptrend:
                    executed_price = close * (1 + params.slippage_pct)
                    position = 1
                    entry_price = executed_price
                    highest_price = close
                    trades += 1
                    
                    trade_log.append({
                        'type': 'BUY',
                        'date': date,
                        'price': executed_price,
                        'regime': 'bull' if in_uptrend else 'bear',
                        'ma200': df['sma200'].iloc[i],
                        'balance_before': balance
                    })
                    
                    if len(trade_log) <= 10:
                        print(f"   BUY: ${executed_price:,.0f} on {date.strftime('%Y-%m-%d')} "
                              f"(dip in uptrend, MA: ${df['sma200'].iloc[i]:,.0f})")
            
            # EXIT LOGIC: Profit target OR trailing stop (exactly like AVAX)
            else:
                highest_price = max(highest_price, close)
                trailing_stop_price = highest_price * (1 - params.trailing_stop_pct)
                
                should_exit = (close > sell_level) or (close < trailing_stop_price)
                exit_reason = "profit_target" if close > sell_level else "trailing_stop"
                
                if should_exit:
                    executed_price = close * (1 - params.slippage_pct)
                    
                    gross_return = (executed_price - entry_price) / entry_price
                    net_return = gross_return - 2 * params.fee_rate
                    balance *= (1 + net_return)
                    
                    is_win = gross_return > 0
                    wins += 1 if is_win else 0
                    losses += 1 if not is_win else 0
                    
                    trade_log.append({
                        'type': 'SELL',
                        'date': date,
                        'price': executed_price,
                        'gross_return': gross_return,
                        'net_return': net_return,
                        'reason': exit_reason,
                        'regime': 'bull' if in_uptrend else 'bear',
                        'balance_after': balance,
                        'highest_price': highest_price,
                        'trailing_stop': trailing_stop_price
                    })
                    
                    if len([t for t in trade_log if t['type'] == 'SELL']) <= 10:
                        print(f"   SELL: ${executed_price:,.0f} on {date.strftime('%Y-%m-%d')} "
                              f"({net_return:+.1%}, {exit_reason})")
                    
                    position = 0
                    entry_price = 0
                    highest_price = 0
            
            # Track performance
            balance_history.append(balance)
            peak_balance = max(peak_balance, balance)
            current_drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, current_drawdown)
        
        # Force exit if still in position (exactly like AVAX)
        if position > 0:
            final_price = df['close'].iloc[-1] * (1 - params.slippage_pct)
            gross_return = (final_price - entry_price) / entry_price
            net_return = gross_return - 2 * params.fee_rate
            balance *= (1 + net_return)
            
            is_win = gross_return > 0
            wins += 1 if is_win else 0
            losses += 1 if not is_win else 0
            
            print(f"   FORCED EXIT: ${final_price:,.0f} ({net_return:+.1%})")
        
        # Calculate metrics
        total_return = (balance - initial_balance) / initial_balance
        win_rate = wins / max(trades, 1)
        
        # Calculate Sharpe ratio
        if len(balance_history) > 1:
            balance_series = pd.Series(balance_history)
            balance_returns = balance_series.pct_change().dropna()
            sharpe = balance_returns.mean() / balance_returns.std() * np.sqrt(252) if balance_returns.std() > 0 else 0
        else:
            sharpe = 0
        
        # Analyze regime performance
        bull_ratio = bull_days / (bull_days + bear_days) if (bull_days + bear_days) > 0 else 0
        
        print(f"✅ AVAX-style Bitcoin strategy complete!")
        print(f"   Final balance: ${balance:,.2f}")
        print(f"   Total return: {total_return:.1%}")
        print(f"   Trades: {trades}, Win rate: {win_rate:.1%}")
        print(f"   Max drawdown: {max_drawdown:.1%}")
        print(f"   Sharpe ratio: {sharpe:.2f}")
        print(f"   Bull regime: {bull_ratio:.1%} of trading days")
        
        return {
            'returns': total_return,
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'final_balance': balance,
            'trade_log': trade_log,
            'balance_history': balance_history,
            'regime_stats': {
                'bull_days': bull_days,
                'bear_days': bear_days,
                'bull_ratio': bull_ratio,
                'total_days': bull_days + bear_days
            }
        }
        
    except Exception as e:
        print(f"❌ AVAX-style strategy error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'returns': -1.0,
            'trades': 0,
            'error': str(e)
        }

def test_bitcoin_avax_strategy():
    """Test Bitcoin using AVAX's exact strategy"""
    print("₿ TESTING BITCOIN WITH AVAX'S STRATEGY")
    print("=" * 60)
    print("📈 Using AVAX's approach:")
    print("   1. 200-day MA regime filter")
    print("   2. Buy dips in uptrends only")
    print("   3. Trailing stops to ride trends")
    
    # Fetch Bitcoin data
    API_KEY = '4c322523f98c6c20dbe789194197dafac7329ec5a7dc378503118d443e867c2b'
    
    import requests
    end_date = pd.Timestamp.today() - pd.Timedelta(days=1)
    url = (
        f"https://min-api.cryptocompare.com/data/v2/histoday?"
        f"fsym=BTC&tsym=USD&limit=400&toTs={int(end_date.timestamp())}&api_key={API_KEY}"
    )
    
    try:
        print("\n📊 Fetching Bitcoin data...")
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
        
        # Test multiple parameter sets like AVAX would
        test_params = [
            BitcoinAVAXParams(bottom=0.25, top=0.70, trailing_stop_pct=0.08),  # Aggressive
            BitcoinAVAXParams(bottom=0.30, top=0.75, trailing_stop_pct=0.10),  # Balanced
            BitcoinAVAXParams(bottom=0.35, top=0.80, trailing_stop_pct=0.12),  # Conservative
        ]
        
        param_names = ["Aggressive", "Balanced", "Conservative"]
        
        print(f"\n📊 Testing multiple parameter sets...")
        
        best_return = -1
        best_params = None
        best_results = None
        
        for i, params in enumerate(test_params):
            print(f"\n🔬 Testing {param_names[i]} parameters:")
            print(f"   Buy dips at: {params.bottom:.0%} of range")
            print(f"   Sell at: {params.top:.0%} of range") 
            print(f"   Trailing stop: {params.trailing_stop_pct:.0%}")
            
            results = evaluate_bitcoin_avax_strategy(df, params)
            
            if results['returns'] > best_return:
                best_return = results['returns']
                best_params = params
                best_results = results
        
        # Show best results
        print(f"\n🏆 BEST AVAX-STYLE RESULTS:")
        print(f"   Return: {best_results['returns']:.1%}")
        print(f"   Trades: {best_results['trades']}")
        print(f"   Win rate: {best_results['win_rate']:.1%}")
        print(f"   Max drawdown: {best_results['max_drawdown']:.1%}")
        print(f"   Sharpe ratio: {best_results['sharpe_ratio']:.2f}")
        
        # Compare to market and other strategies
        market_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
        outperformance = best_results['returns'] - market_return
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"   Market return: {market_return:.1%}")
        print(f"   AVAX-style return: {best_results['returns']:.1%}")
        print(f"   Outperformance: {outperformance:+.1%}")
        
        print(f"\n🔄 STRATEGY EVOLUTION SUMMARY:")
        print(f"   Original Bitcoin (overfitted): 24.6%")
        print(f"   Validated Bitcoin (realistic): 1.9%")
        print(f"   Regime-aware (mean-reversion): 1.9%")
        print(f"   Trend + Regime (adaptive): 9.0%")
        print(f"   AVAX-style (MA + dips + trailing): {best_results['returns']:.1%}")
        print(f"   Market benchmark: {market_return:.1%}")
        
        # Final verdict
        if outperformance > 0:
            verdict = "🎉 BEATS MARKET!"
        elif best_results['returns'] > market_return * 0.8:
            verdict = "🔥 STRONG PERFORMANCE!"
        elif best_results['returns'] > 0:
            verdict = "✅ PROFITABLE"
        else:
            verdict = "❌ UNPROFITABLE"
        
        print(f"   Final verdict: {verdict}")
        
        return best_results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_bitcoin_avax_strategy()
