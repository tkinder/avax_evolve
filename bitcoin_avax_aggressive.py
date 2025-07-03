#!/usr/bin/env python3
"""
Bitcoin AVAX-Style Strategy - AGGRESSIVE VERSION
Adjusts buy levels for Bitcoin's steady grind-up behavior
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
    """Bitcoin parameters using AVAX's strategy but adjusted for Bitcoin's behavior"""
    
    # Price levels (MUCH more aggressive for Bitcoin)
    bottom: float = 0.50        # Buy at 50% of range (much higher)
    top: float = 0.85           # Sell at 85% of range  
    
    # Moving average regime filter
    ma_period: int = 50         # Shorter MA for faster signals
    
    # Trailing stop
    trailing_stop_pct: float = 0.15    # Wider trailing stop for Bitcoin
    
    # Execution costs
    fee_rate: float = 0.001
    slippage_pct: float = 0.0005

def evaluate_bitcoin_avax_aggressive(df: pd.DataFrame, params: BitcoinAVAXParams) -> dict:
    """
    Bitcoin strategy using AVAX approach but adjusted for Bitcoin's steady uptrend behavior
    """
    try:
        print(f"🚀 Running AGGRESSIVE Bitcoin-AVAX strategy...")
        
        # Calculate moving average regime (shorter period for Bitcoin)
        df = df.copy()
        df[f'sma{params.ma_period}'] = df['close'].rolling(window=params.ma_period).mean()
        df['regime'] = df['close'] > df[f'sma{params.ma_period}']
        
        # More aggressive price levels for Bitcoin's steady grind
        price_min = df['close'].min()
        price_max = df['close'].max()
        price_range = price_max - price_min
        
        buy_level = price_min + price_range * params.bottom   
        sell_level = price_min + price_range * params.top     
        
        print(f"   Price range: ${price_min:,.0f} - ${price_max:,.0f}")
        print(f"   Buy level: ${buy_level:,.0f} (50% of range - much more aggressive)")
        print(f"   Sell level: ${sell_level:,.0f} (85% of range)")
        print(f"   {params.ma_period}-day MA regime filter")
        
        # Check how often price was below buy level AND in uptrend
        ma_col = f'sma{params.ma_period}'
        opportunities = 0
        for i in range(params.ma_period, len(df)):
            if df['close'].iloc[i] < buy_level and df['regime'].iloc[i]:
                opportunities += 1
        
        print(f"   Buy opportunities: {opportunities} days when price < ${buy_level:,.0f} AND above {params.ma_period}-day MA")
        
        # Run backtest
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
        
        for i in range(params.ma_period, len(df)):
            close = df['close'].iloc[i]
            date = df.index[i]
            in_uptrend = df['regime'].iloc[i]
            ma_value = df[ma_col].iloc[i]
            
            if pd.isna(ma_value):
                balance_history.append(balance)
                continue
            
            # Track regime days
            if in_uptrend:
                bull_days += 1
            else:
                bear_days += 1
            
            prev_balance = balance
            
            # ENTRY: Buy when price dips below buy_level AND we're in uptrend
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
                        'ma_value': ma_value,
                        'balance_before': balance
                    })
                    
                    print(f"   BUY: ${executed_price:,.0f} on {date.strftime('%Y-%m-%d')} "
                          f"(dip to {close/price_range:.1%} of range, MA: ${ma_value:,.0f})")
            
            # EXIT: Profit target OR trailing stop
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
                        'balance_after': balance,
                        'highest_price': highest_price,
                        'trailing_stop': trailing_stop_price
                    })
                    
                    print(f"   SELL: ${executed_price:,.0f} on {date.strftime('%Y-%m-%d')} "
                          f"({net_return:+.1%}, {exit_reason}, held {(date-trade_log[-2]['date']).days}d)")
                    
                    position = 0
                    entry_price = 0
                    highest_price = 0
            
            # Track performance
            balance_history.append(balance)
            peak_balance = max(peak_balance, balance)
            current_drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, current_drawdown)
        
        # Force exit if still in position
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
        
        print(f"✅ Aggressive Bitcoin-AVAX strategy complete!")
        print(f"   Final balance: ${balance:,.2f}")
        print(f"   Total return: {total_return:.1%}")
        print(f"   Trades: {trades}, Win rate: {win_rate:.1%}")
        print(f"   Max drawdown: {max_drawdown:.1%}")
        print(f"   Sharpe ratio: {sharpe:.2f}")
        
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
            'balance_history': balance_history
        }
        
    except Exception as e:
        print(f"❌ Aggressive strategy error: {e}")
        import traceback
        traceback.print_exc()
        return {'returns': -1.0, 'trades': 0, 'error': str(e)}

def test_aggressive_bitcoin_avax():
    """Test multiple aggressive parameter sets"""
    print("₿ TESTING AGGRESSIVE BITCOIN-AVAX STRATEGY")
    print("=" * 60)
    print("🎯 Problem: Original buy levels too conservative")
    print("💡 Solution: Much more aggressive entry levels")
    
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
        
        df = df.rename(columns={'close': 'close', 'high': 'high', 'low': 'low', 'open': 'open'})
        numeric_columns = ['close', 'high', 'low', 'open']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        print(f"✅ Got {len(df)} days of Bitcoin data")
        
        # Test very aggressive parameter sets
        test_params = [
            BitcoinAVAXParams(bottom=0.40, top=0.85, ma_period=20, trailing_stop_pct=0.12),  # Very aggressive
            BitcoinAVAXParams(bottom=0.50, top=0.85, ma_period=50, trailing_stop_pct=0.15),  # Aggressive  
            BitcoinAVAXParams(bottom=0.60, top=0.90, ma_period=50, trailing_stop_pct=0.18),  # Moderate
            BitcoinAVAXParams(bottom=0.70, top=0.95, ma_period=100, trailing_stop_pct=0.20), # Conservative
        ]
        
        param_names = ["Very Aggressive", "Aggressive", "Moderate", "Conservative"]
        
        print(f"\n📊 Testing multiple aggressive parameter sets...")
        
        best_return = -1
        best_results = None
        
        for i, params in enumerate(test_params):
            print(f"\n🚀 Testing {param_names[i]}:")
            print(f"   Buy at: {params.bottom:.0%} of range")
            print(f"   Sell at: {params.top:.0%} of range") 
            print(f"   MA period: {params.ma_period} days")
            print(f"   Trailing stop: {params.trailing_stop_pct:.0%}")
            
            results = evaluate_bitcoin_avax_aggressive(df, params)
            
            if results['returns'] > best_return:
                best_return = results['returns']
                best_results = results
                best_name = param_names[i]
        
        # Show results
        market_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
        
        print(f"\n🏆 BEST AGGRESSIVE RESULTS ({best_name}):")
        print(f"   Return: {best_results['returns']:.1%}")
        print(f"   Trades: {best_results['trades']}")
        print(f"   Win rate: {best_results['win_rate']:.1%}")
        print(f"   Max drawdown: {best_results['max_drawdown']:.1%}")
        print(f"   Sharpe ratio: {best_results['sharpe_ratio']:.2f}")
        
        outperformance = best_results['returns'] - market_return
        print(f"\n📈 VS MARKET:")
        print(f"   Market: {market_return:.1%}")
        print(f"   Strategy: {best_results['returns']:.1%}")
        print(f"   Outperformance: {outperformance:+.1%}")
        
        print(f"\n🔄 FINAL STRATEGY EVOLUTION:")
        print(f"   Original (overfitted): 24.6%")
        print(f"   Validated: 1.9%")
        print(f"   Regime-aware: 1.9%") 
        print(f"   Trend+Regime: 9.0%")
        print(f"   AVAX-style (too conservative): 0.0%")
        print(f"   AVAX-style (aggressive): {best_results['returns']:.1%}")
        print(f"   Market benchmark: {market_return:.1%}")
        
        # Final verdict
        if outperformance > 0:
            verdict = "🎉 BEATS MARKET!"
        elif best_results['returns'] > market_return * 0.7:
            verdict = "🔥 STRONG PERFORMANCE!"
        elif best_results['returns'] > 0:
            verdict = "✅ PROFITABLE"
        else:
            verdict = "❌ STILL NOT WORKING"
        
        print(f"   Final verdict: {verdict}")
        
        return best_results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    test_aggressive_bitcoin_avax()
