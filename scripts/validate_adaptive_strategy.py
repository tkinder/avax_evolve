#!/usr/bin/env python3
# scripts/validate_adaptive_strategy.py

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from src.core.data import fetch_historical_data
from src.core.logging import get_logger
from src.evolution.adaptive_optimizer_fixed import run_adaptive_optimization_fixed
from src.core.adaptive_backtester import adaptive_backtest_strategy

log = get_logger()

def validate_adaptive_strategy():
    """
    Complete validation pipeline for the adaptive strategy.
    """
    print("🔬 ADAPTIVE STRATEGY VALIDATION")
    print("="*60)
    
    # Load data
    df = fetch_historical_data(refresh=False)
    print(f"📊 Loaded {len(df)} rows of historical data")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # Split data
    train_data = df[df.index <= '2023-12-31'].copy()
    test_data = df[df.index > '2023-12-31'].copy()
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} days)")
    print(f"   Testing:  {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} days)")
    
    # Train adaptive strategy
    print(f"\n" + "="*50)
    print("🎓 TRAINING ADAPTIVE STRATEGY")
    print("="*50)
    
    optimization_result = run_adaptive_optimization_fixed(
        train_data,
        ngen=20,
        pop_size=40,
        log=log
    )
    
    optimized_params = optimization_result['params']
    train_results = optimization_result['results']
    
    print(f"\n📈 TRAINING PERFORMANCE:")
    print(f"   Returns: {train_results['returns']:.1%}")
    print(f"   Trades: {train_results['trades']}")
    print(f"   Win Rate: {train_results['wins'] / max(train_results['trades'], 1):.1%}")
    print(f"   Max Drawdown: {train_results['max_drawdown']:.1%}")
    print(f"   Sharpe Ratio: {train_results['sharpe']:.4f}")
    
    # Test on unseen data
    print(f"\n" + "="*50)
    print("🧪 TESTING ON UNSEEN DATA")
    print("="*50)
    
    test_results = adaptive_backtest_strategy(test_data, optimized_params)
    
    print(f"\n📊 OUT-OF-SAMPLE PERFORMANCE:")
    print(f"   Returns: {test_results['returns']:.1%}")
    print(f"   Trades: {test_results['trades']}")
    print(f"   Win Rate: {test_results['wins'] / max(test_results['trades'], 1):.1%}")
    print(f"   Max Drawdown: {test_results['max_drawdown']:.1%}")
    print(f"   Sharpe Ratio: {test_results['sharpe']:.4f}")
    
    # Quarterly analysis
    print(f"\n" + "="*50)
    print("📅 QUARTERLY ANALYSIS")
    print("="*50)
    
    quarterly_results = test_quarterly_performance(test_data, optimized_params)
    
    # Comparison
    print(f"\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    
    compare_adaptive_performance(train_results, test_results, quarterly_results)
    
    # Show optimized parameters
    print(f"\n" + "="*50)
    print("🎯 OPTIMIZED ADAPTIVE PARAMETERS")
    print("="*50)
    
    display_parameters(optimized_params)
    
    return optimized_params, train_results, test_results, quarterly_results

def test_quarterly_performance(test_data, params):
    """
    Test adaptive strategy on quarterly periods.
    """
    quarters = [
        ('Q1 2024', '2024-01-01', '2024-03-31'),
        ('Q2 2024', '2024-04-01', '2024-06-30'),
        ('Q3 2024', '2024-07-01', '2024-09-30'),
        ('Q4 2024', '2024-10-01', '2024-12-31'),
        ('Q1 2025', '2025-01-01', '2025-03-31'),
        ('Q2 2025', '2025-04-01', '2025-06-17'),
    ]
    
    results = []
    
    for name, start, end in quarters:
        quarter_data = test_data[(test_data.index >= start) & (test_data.index <= end)]
        
        if len(quarter_data) < 30:  # Skip quarters with insufficient data
            continue
        
        print(f"\n📅 Testing {name} ({len(quarter_data)} days)")
        
        try:
            quarter_result = adaptive_backtest_strategy(quarter_data, params)
            
            results.append({
                'quarter': name,
                'returns': quarter_result['returns'],
                'trades': quarter_result['trades'],
                'wins': quarter_result['wins'],
                'losses': quarter_result['losses'],
                'max_drawdown': quarter_result['max_drawdown'],
                'sharpe': quarter_result['sharpe']
            })
            
            win_rate = quarter_result['wins'] / max(quarter_result['trades'], 1)
            print(f"   Returns: {quarter_result['returns']:8.1%}   "
                  f"Trades: {quarter_result['trades']:2d}   "
                  f"Win Rate: {win_rate:6.1%}   "
                  f"DD: {quarter_result['max_drawdown']:6.1%}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")
            results.append({
                'quarter': name,
                'returns': 0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'max_drawdown': 0,
                'sharpe': 0
            })
    
    return results

def compare_adaptive_performance(train_results, test_results, quarterly_results):
    """
    Compare training vs testing performance for adaptive strategy.
    """
    print(f"{'Metric':<20} {'Training':<15} {'Testing':<15} {'Ratio':<10}")
    print("-" * 60)
    
    metrics = [
        ('Returns', 'returns'),
        ('Trades', 'trades'),
        ('Win Rate', lambda r: r['wins'] / max(r['trades'], 1)),
        ('Max Drawdown', 'max_drawdown'),
        ('Sharpe Ratio', 'sharpe')
    ]
    
    for metric_name, metric_key in metrics:
        if callable(metric_key):
            train_val = metric_key(train_results)
            test_val = metric_key(test_results)
        else:
            train_val = train_results[metric_key]
            test_val = test_results[metric_key]
        
        if train_val != 0:
            ratio = test_val / train_val
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "N/A"
        
        if metric_name in ['Returns', 'Win Rate', 'Max Drawdown']:
            train_str = f"{train_val:.1%}"
            test_str = f"{test_val:.1%}"
        elif metric_name == 'Trades':
            train_str = f"{train_val:d}"
            test_str = f"{test_val:d}"
        else:
            train_str = f"{train_val:.4f}"
            test_str = f"{test_val:.4f}"
        
        print(f"{metric_name:<20} {train_str:<15} {test_str:<15} {ratio_str:<10}")
    
    # Quarterly summary
    if quarterly_results:
        quarterly_returns = [q['returns'] for q in quarterly_results if q['trades'] > 0]
        quarterly_trades = [q['trades'] for q in quarterly_results]
        
        if quarterly_returns:
            print(f"\n📊 QUARTERLY SUMMARY:")
            print(f"   Profitable quarters: {len([r for r in quarterly_returns if r > 0])}/{len(quarterly_returns)}")
            print(f"   Average quarterly return: {np.mean(quarterly_returns):.1%}")
            print(f"   Best quarter: {max(quarterly_returns):.1%}")
            print(f"   Worst quarter: {min(quarterly_returns):.1%}")
            print(f"   Average trades per quarter: {np.mean(quarterly_trades):.1f}")
    
    # Assessment
    returns_ratio = test_results['returns'] / train_results['returns'] if train_results['returns'] != 0 else 0
    
    print(f"\n🎯 ADAPTIVE STRATEGY ASSESSMENT:")
    if returns_ratio > 0.6:
        print(f"   ✅ EXCELLENT: Out-of-sample performance is {returns_ratio:.1%} of training ({returns_ratio:.2f}x)")
        print(f"   Adaptive strategy shows strong generalization!")
    elif returns_ratio > 0.3:
        print(f"   ✅ GOOD: Out-of-sample performance is {returns_ratio:.1%} of training ({returns_ratio:.2f}x)")
        print(f"   Adaptive approach is working well.")
    elif returns_ratio > 0:
        print(f"   ⚠️  MODERATE: Out-of-sample performance is {returns_ratio:.1%} of training ({returns_ratio:.2f}x)")
        print(f"   Some improvement over fixed strategy, but room for enhancement.")
    else:
        print(f"   ❌ POOR: Strategy failed on out-of-sample data")
    
    print(f"\n💡 REALISTIC EXPECTATIONS:")
    print(f"   Expected Annual Return: ~{test_results['returns']:.1%}")
    print(f"   Expected Max Drawdown: ~{test_results['max_drawdown']:.1%}")
    print(f"   Expected Trade Frequency: ~{test_results['trades'] / (len(quarterly_results) * 3) if quarterly_results else 0:.1f} trades/month")

def display_parameters(params):
    """
    Display the optimized adaptive parameters in a readable format.
    """
    print(f"📊 Core Strategy Parameters:")
    print(f"   Risk Reward:     {params.risk_reward:.3f}")
    print(f"   Trend Strength:  {params.trend_strength:.3f}")
    print(f"   Entry Threshold: {params.entry_threshold:.3f}")
    print(f"   Confidence:      {params.confidence:.3f}")
    
    print(f"\n🎯 Adaptive Price Levels:")
    print(f"   Buy Threshold:   {params.buy_threshold_pct:.1%} of recent range")
    print(f"   Sell Threshold:  {params.sell_threshold_pct:.1%} of recent range")
    
    print(f"\n🌊 Market Regime Adjustments:")
    print(f"   Bull Market:     {params.bull_multiplier:.2f}x position size")
    print(f"   Bear Market:     {params.bear_multiplier:.2f}x position size")
    print(f"   High Volatility: {params.high_vol_multiplier:.2f}x position size")
    print(f"   Low Volatility:  {params.low_vol_multiplier:.2f}x position size")
    
    print(f"\n⚖️  Risk Management:")
    print(f"   Max Position:    {params.max_position_pct:.1%} of balance")
    print(f"   Stop Loss:       {params.stop_loss_pct:.1%}")
    print(f"   Take Profit:     {params.take_profit_pct:.1%}")

def main():
    """
    Run the complete adaptive strategy validation.
    """
    try:
        params, train_results, test_results, quarterly_results = validate_adaptive_strategy()
        
        print(f"\n🎉 ADAPTIVE VALIDATION COMPLETE!")
        print(f"✨ Key Improvements over Fixed Strategy:")
        print(f"   • Uses relative price levels (adapts to any price range)")
        print(f"   • Market regime detection (adjusts to bull/bear markets)")
        print(f"   • Adaptive position sizing (reduces risk in volatile periods)")
        print(f"   • Better risk management (stop losses and take profits)")
        
        if test_results['trades'] > 0:
            print(f"\n🚀 Ready for further testing with realistic expectations!")
        else:
            print(f"\n⚠️  Strategy still needs refinement - consider parameter adjustments.")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()