#!/usr/bin/env python3
# scripts/rolling_test.py - Simplified Rolling Validation

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
from datetime import datetime
from src.core.data import fetch_historical_data
from src.core.logging import get_logger
from src.evolution.phase1_fixed import run_phase1_optimization
from src.evolution.phase2_fixed import run_phase2_optimization
from src.core.backtester import backtest_strategy

log = get_logger()

def create_test_periods(df, train_end='2023-12-31'):
    """
    Create quarterly test periods for 2024-2025.
    """
    train_data = df[df.index <= train_end].copy()
    
    # Define test periods (quarters)
    test_periods = [
        ('Q1 2024', '2024-01-01', '2024-03-31'),
        ('Q2 2024', '2024-04-01', '2024-06-30'), 
        ('Q3 2024', '2024-07-01', '2024-09-30'),
        ('Q4 2024', '2024-10-01', '2024-12-31'),
        ('Q1 2025', '2025-01-01', '2025-03-31'),
        ('Q2 2025', '2025-04-01', '2025-06-17'),  # Up to current date
    ]
    
    periods_data = []
    for name, start, end in test_periods:
        period_data = df[(df.index >= start) & (df.index <= end)].copy()
        if len(period_data) > 10:  # Only include periods with sufficient data
            periods_data.append({
                'name': name,
                'start': start,
                'end': end, 
                'data': period_data,
                'days': len(period_data)
            })
    
    print(f"📊 Training data: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} days)")
    print(f"🧪 Test periods created: {len(periods_data)} quarters")
    
    return train_data, periods_data

def optimize_once_on_training_data(train_data):
    """
    Optimize strategy parameters once using 2023 training data.
    We'll use these same parameters for all quarterly tests.
    """
    print("\n" + "="*50)
    print("🎓 OPTIMIZING ON 2023 TRAINING DATA")
    print("="*50)
    
    # Phase 1 optimization
    print("⚙️  Running Phase 1 optimization...")
    p1_result = run_phase1_optimization(
        train_data, 
        ngen=15,      # More thorough optimization
        pop_size=40,  
        log=None      # Suppress detailed logging for cleaner output
    )
    
    # Phase 2 optimization  
    print("⚙️  Running Phase 2 optimization...")
    p2_result = run_phase2_optimization(
        train_data, 
        base_params=p1_result['params'], 
        ngen=20,      
        pop_size=50,  
        log=None
    )
    
    optimized_params = p2_result['params']
    
    # Test on training data for reference
    train_backtest = backtest_strategy(train_data, optimized_params)
    
    print(f"\n📈 TRAINING PERFORMANCE (2023 data):")
    print(f"   Returns: {train_backtest['returns']:.1%}")
    print(f"   Sharpe: {train_backtest.get('sharpe', 0):.4f}")
    print(f"   Max Drawdown: {train_backtest['max_drawdown']:.1%}")
    print(f"   Volatility: {train_backtest['volatility']:.1%}")
    
    print(f"\n🎯 Optimized Parameters:")
    print(f"   {optimized_params}")
    
    return optimized_params, train_backtest

def test_on_quarters(periods_data, optimized_params):
    """
    Test the same optimized parameters on each quarter.
    """
    print("\n" + "="*50)
    print("🧪 TESTING ON QUARTERLY PERIODS")
    print("="*50)
    
    results = []
    
    for period in periods_data:
        print(f"\n📅 Testing {period['name']} ({period['start']} to {period['end']}, {period['days']} days)")
        
        try:
            # Run backtest on this quarter
            backtest_result = backtest_strategy(period['data'], optimized_params)
            
            result = {
                'period': period['name'],
                'start': period['start'],
                'end': period['end'],
                'days': period['days'],
                'returns': backtest_result['returns'],
                'sharpe': backtest_result.get('sharpe', 0),
                'max_drawdown': backtest_result['max_drawdown'],
                'volatility': backtest_result['volatility'],
                'status': 'success'
            }
            
            print(f"   Returns: {result['returns']:8.1%}   Drawdown: {result['max_drawdown']:6.1%}   Sharpe: {result['sharpe']:6.4f}")
            
        except Exception as e:
            result = {
                'period': period['name'],
                'start': period['start'], 
                'end': period['end'],
                'days': period['days'],
                'returns': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'status': f'error: {str(e)[:50]}'
            }
            
            print(f"   ❌ Error: {str(e)[:50]}")
        
        results.append(result)
    
    return results

def analyze_quarterly_results(results, train_performance):
    """
    Analyze and summarize the quarterly test results.
    """
    print("\n" + "="*60)
    print("📊 QUARTERLY PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Filter successful results
    successful_results = [r for r in results if r['status'] == 'success']
    
    if not successful_results:
        print("❌ No successful quarterly tests found!")
        return
    
    # Extract metrics
    quarterly_returns = [r['returns'] for r in successful_results]
    quarterly_drawdowns = [r['max_drawdown'] for r in successful_results]
    quarterly_sharpes = [r['sharpe'] for r in successful_results]
    
    # Annualize quarterly returns (roughly)
    annualized_returns = [(1 + r) ** (365 / 90) - 1 for r in quarterly_returns]
    
    print(f"📈 QUARTERLY RESULTS SUMMARY:")
    print(f"   Successful periods: {len(successful_results)}/{len(results)}")
    print(f"   Profitable quarters: {len([r for r in quarterly_returns if r > 0])}/{len(successful_results)}")
    print(f"   Win rate: {len([r for r in quarterly_returns if r > 0]) / len(successful_results):.1%}")
    
    print(f"\n📊 QUARTERLY RETURNS:")
    print(f"   Average: {np.mean(quarterly_returns):8.1%}")
    print(f"   Median:  {np.median(quarterly_returns):8.1%}")
    print(f"   Std Dev: {np.std(quarterly_returns):8.1%}")
    print(f"   Best:    {max(quarterly_returns):8.1%}")
    print(f"   Worst:   {min(quarterly_returns):8.1%}")
    
    print(f"\n📊 ANNUALIZED ESTIMATES (from quarterly data):")
    print(f"   Average annual: {np.mean(annualized_returns):8.1%}")
    print(f"   Median annual:  {np.median(annualized_returns):8.1%}")
    print(f"   Best annual:    {max(annualized_returns):8.1%}")
    print(f"   Worst annual:   {min(annualized_returns):8.1%}")
    
    print(f"\n📊 RISK METRICS:")
    print(f"   Average drawdown: {np.mean(quarterly_drawdowns):6.1%}")
    print(f"   Max drawdown:     {max(quarterly_drawdowns):6.1%}")
    print(f"   Average Sharpe:   {np.mean(quarterly_sharpes):6.4f}")
    
    # Detailed quarterly breakdown
    print(f"\n📅 DETAILED QUARTERLY BREAKDOWN:")
    print(f"{'Period':<12} {'Returns':<10} {'Drawdown':<10} {'Sharpe':<8} {'Ann. Est.':<10}")
    print("-" * 60)
    
    for i, result in enumerate(successful_results):
        ann_est = annualized_returns[i]
        print(f"{result['period']:<12} {result['returns']:>8.1%} {result['max_drawdown']:>8.1%} "
              f"{result['sharpe']:>6.4f} {ann_est:>8.1%}")
    
    # Comparison with training
    train_return = train_performance['returns']
    avg_quarterly = np.mean(quarterly_returns)
    
    print(f"\n🔍 COMPARISON WITH TRAINING:")
    print(f"   Training (2023):     {train_return:8.1%}")
    print(f"   Avg Quarterly Test:  {avg_quarterly:8.1%}")
    print(f"   Degradation:         {avg_quarterly/train_return:8.2f}x")
    
    # Realistic expectations
    median_annual = np.median(annualized_returns)
    print(f"\n💡 REALISTIC LIVE TRADING EXPECTATIONS:")
    print(f"   Expected annual return: {median_annual:8.1%}")
    print(f"   Expected max drawdown:  {max(quarterly_drawdowns):8.1%}")
    print(f"   Probability of profit:  {len([r for r in quarterly_returns if r > 0]) / len(successful_results):.0%}")
    
    return {
        'quarterly_returns': quarterly_returns,
        'annualized_estimate': median_annual,
        'max_drawdown': max(quarterly_drawdowns),
        'win_rate': len([r for r in quarterly_returns if r > 0]) / len(successful_results)
    }

def main():
    """
    Main simplified rolling test pipeline.
    """
    print("🔄 SIMPLIFIED ROLLING TEST ANALYSIS")
    print("="*60)
    
    # Load data
    df = fetch_historical_data(refresh=False)
    print(f"📊 Loaded {len(df)} rows of historical data")
    
    # Create test periods
    train_data, periods_data = create_test_periods(df)
    
    # Optimize once on training data
    optimized_params, train_performance = optimize_once_on_training_data(train_data)
    
    # Test on all quarters using same parameters
    quarterly_results = test_on_quarters(periods_data, optimized_params)
    
    # Analyze results
    summary = analyze_quarterly_results(quarterly_results, train_performance)
    
    print(f"\n🎉 ROLLING TEST COMPLETE!")
    print(f"💡 Key Takeaway: Expect ~{summary['annualized_estimate']:.1%} annual returns with {summary['max_drawdown']:.1%} max drawdown")

if __name__ == "__main__":
    main()