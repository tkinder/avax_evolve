#!/usr/bin/env python3
# scripts/validate_strategy.py - Out-of-Sample Testing

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

def split_data_for_validation(df, train_end_date='2023-12-31'):
    """
    Split data into training (in-sample) and testing (out-of-sample) periods.
    
    Args:
        df: Full historical dataset
        train_end_date: Last date to include in training data
        
    Returns:
        train_data, test_data
    """
    print(f"📊 Full dataset: {df.index[0]} to {df.index[-1]} ({len(df)} days)")
    
    # Split the data
    train_data = df[df.index <= train_end_date].copy()
    test_data = df[df.index > train_end_date].copy()
    
    print(f"🎓 Training data: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} days)")
    print(f"🧪 Testing data: {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} days)")
    
    # Validate we have enough data
    if len(train_data) < 200:
        raise ValueError(f"Training data too short: {len(train_data)} days. Need at least 200 days.")
    
    if len(test_data) < 50:
        raise ValueError(f"Testing data too short: {len(test_data)} days. Need at least 50 days.")
    
    return train_data, test_data

def optimize_on_training_data(train_data):
    """
    Optimize strategy parameters using ONLY training data.
    """
    print("\n" + "="*50)
    print("🎓 TRAINING PHASE - Optimizing on Historical Data")
    print("="*50)
    
    log.info("=== Phase 1 Optimization (Training Data Only) ===")
    p1_result = run_phase1_optimization(
        train_data, 
        ngen=12,      # Slightly more generations for better training
        pop_size=35,  
        log=log
    )
    
    train_params1 = p1_result['params']
    train_fitness1 = p1_result['fitness']
    log.info(f"Phase1 Training Result: {train_params1} (fitness={train_fitness1:.4f})")
    
    log.info("=== Phase 2 Optimization (Training Data Only) ===")
    p2_result = run_phase2_optimization(
        train_data, 
        base_params=train_params1, 
        ngen=18,      # Slightly more generations
        pop_size=45,  
        log=log
    )
    
    train_params2 = p2_result['params']
    train_fitness2 = p2_result['fitness']
    log.info(f"Phase2 Training Result: {train_params2} (fitness={train_fitness2:.4f})")
    
    # Test optimized parameters on training data (in-sample performance)
    train_backtest = backtest_strategy(train_data, train_params2)
    
    print(f"\n📈 IN-SAMPLE PERFORMANCE (Training Data):")
    print(f"   Period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"   Returns: {train_backtest['returns']:.1%}")
    print(f"   Sharpe: {train_backtest.get('sharpe', 'N/A'):.4f}")
    print(f"   Max Drawdown: {train_backtest['max_drawdown']:.1%}")
    print(f"   Volatility: {train_backtest['volatility']:.1%}")
    
    return train_params2, {
        'returns': train_backtest['returns'],
        'sharpe': train_backtest.get('sharpe', 0),
        'max_drawdown': train_backtest['max_drawdown'],
        'volatility': train_backtest['volatility']
    }

def test_on_unseen_data(test_data, optimized_params):
    """
    Test the optimized parameters on completely unseen data.
    """
    print("\n" + "="*50)
    print("🧪 TESTING PHASE - Validating on Unseen Data")
    print("="*50)
    
    print(f"Testing parameters on unseen data: {optimized_params}")
    
    # Run backtest on test data using optimized parameters
    test_backtest = backtest_strategy(test_data, optimized_params)
    
    print(f"\n📊 OUT-OF-SAMPLE PERFORMANCE (Unseen Data):")
    print(f"   Period: {test_data.index[0]} to {test_data.index[-1]}")
    print(f"   Returns: {test_backtest['returns']:.1%}")
    print(f"   Sharpe: {test_backtest.get('sharpe', 'N/A'):.4f}")
    print(f"   Max Drawdown: {test_backtest['max_drawdown']:.1%}")
    print(f"   Volatility: {test_backtest['volatility']:.1%}")
    
    return {
        'returns': test_backtest['returns'],
        'sharpe': test_backtest.get('sharpe', 0),
        'max_drawdown': test_backtest['max_drawdown'],
        'volatility': test_backtest['volatility']
    }

def compare_performance(train_metrics, test_metrics):
    """
    Compare in-sample vs out-of-sample performance.
    """
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    
    print(f"{'Metric':<20} {'Training (In-Sample)':<25} {'Testing (Out-of-Sample)':<25} {'Ratio':<10}")
    print("-" * 80)
    
    metrics = ['returns', 'sharpe', 'max_drawdown', 'volatility']
    
    for metric in metrics:
        train_val = train_metrics[metric]
        test_val = test_metrics[metric]
        
        # Calculate ratio (test/train)
        if train_val != 0:
            ratio = test_val / train_val
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "N/A"
        
        # Format values
        if metric in ['returns', 'max_drawdown', 'volatility']:
            train_str = f"{train_val:.1%}"
            test_str = f"{test_val:.1%}"
        else:
            train_str = f"{train_val:.4f}"
            test_str = f"{test_val:.4f}"
        
        print(f"{metric.title():<20} {train_str:<25} {test_str:<25} {ratio_str:<10}")
    
    # Performance assessment
    returns_ratio = test_metrics['returns'] / train_metrics['returns'] if train_metrics['returns'] != 0 else 0
    
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    
    if returns_ratio > 0.7:
        print(f"   ✅ GOOD: Out-of-sample returns are {returns_ratio:.1%} of in-sample ({returns_ratio:.2f}x)")
        print(f"   Strategy shows good generalization to unseen data.")
    elif returns_ratio > 0.3:
        print(f"   ⚠️  MODERATE: Out-of-sample returns are {returns_ratio:.1%} of in-sample ({returns_ratio:.2f}x)")
        print(f"   Strategy works but performance degrades on unseen data.")
    else:
        print(f"   ❌ POOR: Out-of-sample returns are {returns_ratio:.1%} of in-sample ({returns_ratio:.2f}x)")
        print(f"   Strategy may be overfitted to training data.")
    
    # Realistic expectations
    test_annual_return = test_metrics['returns']
    print(f"\n💡 REALISTIC EXPECTATIONS FOR LIVE TRADING:")
    print(f"   Expected Annual Return: ~{test_annual_return:.1%}")
    print(f"   Expected Max Drawdown: ~{test_metrics['max_drawdown']:.1%}")
    print(f"   Risk-Adjusted Return (Sharpe): {test_metrics['sharpe']:.4f}")

def walk_forward_analysis(df, window_months=6, step_months=3):
    """
    Advanced validation: Test strategy across multiple time periods.
    """
    print("\n" + "="*60)
    print("🚶 WALK-FORWARD ANALYSIS")
    print("="*60)
    
    results = []
    
    # Convert to monthly periods
    start_date = df.index[0]
    end_date = df.index[-1]
    
    current_date = start_date
    window_count = 0
    
    while current_date < end_date - pd.DateOffset(months=window_months + step_months):
        window_count += 1
        
        # Define training window
        train_start = current_date
        train_end = current_date + pd.DateOffset(months=window_months)
        
        # Define testing window  
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(months=step_months)
        
        # Skip if not enough data
        if test_end > end_date:
            break
            
        print(f"\n📅 Window {window_count}:")
        print(f"   Train: {train_start.date()} to {train_end.date()}")
        print(f"   Test:  {test_start.date()} to {test_end.date()}")
        
        try:
            # Get data windows
            train_window = df[(df.index >= train_start) & (df.index <= train_end)]
            test_window = df[(df.index >= test_start) & (df.index <= test_end)]
            
            if len(train_window) < 100 or len(test_window) < 20:
                print(f"   ⚠️  Skipping: insufficient data")
                current_date += pd.DateOffset(months=step_months)
                continue
            
            # Quick optimization (reduced parameters for speed)
            p1 = run_phase1_optimization(train_window, ngen=8, pop_size=25, log=None)
            p2 = run_phase2_optimization(train_window, p1['params'], ngen=10, pop_size=30, log=None)
            
            # Test on window
            test_result = backtest_strategy(test_window, p2['params'])
            
            results.append({
                'window': window_count,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'returns': test_result['returns'],
                'sharpe': test_result.get('sharpe', 0),
                'max_drawdown': test_result['max_drawdown']
            })
            
            print(f"   Returns: {test_result['returns']:.1%}, DD: {test_result['max_drawdown']:.1%}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        current_date += pd.DateOffset(months=step_months)
    
    if results:
        # Summary statistics
        returns_list = [r['returns'] for r in results]
        drawdowns_list = [r['max_drawdown'] for r in results]
        
        print(f"\n📊 WALK-FORWARD SUMMARY ({len(results)} windows):")
        print(f"   Average Returns: {np.mean(returns_list):.1%}")
        print(f"   Std Dev Returns: {np.std(returns_list):.1%}")
        print(f"   Win Rate: {len([r for r in returns_list if r > 0]) / len(returns_list):.1%}")
        print(f"   Best Period: {max(returns_list):.1%}")
        print(f"   Worst Period: {min(returns_list):.1%}")
        print(f"   Average Max DD: {np.mean(drawdowns_list):.1%}")
        
        return results
    else:
        print("   ❌ No valid windows found")
        return []

def main():
    """
    Main validation pipeline.
    """
    print("🔬 STRATEGY VALIDATION SYSTEM")
    print("="*60)
    
    # Load full dataset
    df = fetch_historical_data(refresh=False)
    print(f"📊 Loaded {len(df)} rows of historical data")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # 1. Basic Train/Test Split
    train_data, test_data = split_data_for_validation(df, train_end_date='2023-12-31')
    
    # 2. Optimize on training data only
    optimized_params, train_metrics = optimize_on_training_data(train_data)
    
    # 3. Test on completely unseen data
    test_metrics = test_on_unseen_data(test_data, optimized_params)
    
    # 4. Compare performance
    compare_performance(train_metrics, test_metrics)
    
    # 5. Optional: Walk-forward analysis (comment out if too slow)
    print(f"\n❓ Run walk-forward analysis? (This will take 15-30 minutes)")
    response = input("Enter 'y' to run walk-forward analysis, or any other key to skip: ")
    
    if response.lower() == 'y':
        walk_forward_results = walk_forward_analysis(df, window_months=6, step_months=2)
    
    print(f"\n🎉 VALIDATION COMPLETE!")
    print(f"💡 Use the 'Testing (Out-of-Sample)' performance numbers for realistic expectations.")

if __name__ == "__main__":
    main()