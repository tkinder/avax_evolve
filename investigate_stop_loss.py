#!/usr/bin/env python3
"""
Diagnostic script to investigate the -2.11% stop loss parameter issue
"""

import sys
sys.path.append('/home/tkinder/python_projects/avax_evolve')

from bitcoin_backtester import BitcoinParams, evaluate_bitcoin_strategy_with_backtest
from bitcoin_data_pipeline import BitcoinDataPipeline

def investigate_stop_loss_issue():
    print("🔍 INVESTIGATING STOP LOSS PARAMETER ISSUE")
    print("=" * 50)
    
    # Test with the problematic parameters
    print("📊 Testing with reported parameters...")
    
    # Recreate the problematic parameters
    problem_params = BitcoinParams(
        buy_threshold_pct=0.279,     # 27.9%
        sell_threshold_pct=0.858,    # 85.8%
        stop_loss_pct=-0.0211,       # -2.11% (the issue!)
        take_profit_pct=0.179,       # 17.9%
        max_position_pct=1.992       # 199.2%
    )
    
    print(f"🔧 Problematic Parameters:")
    print(f"   Buy Threshold: {problem_params.buy_threshold_pct:.1%}")
    print(f"   Sell Threshold: {problem_params.sell_threshold_pct:.1%}")
    print(f"   Stop Loss: {problem_params.stop_loss_pct:.2%} ⚠️ NEGATIVE!")
    print(f"   Take Profit: {problem_params.take_profit_pct:.1%}")
    print(f"   Max Position: {problem_params.max_position_pct:.1%}")
    
    # Test what happens with negative stop loss
    print(f"\n🧪 Testing negative stop loss behavior...")
    
    # Get Bitcoin data
    pipeline = BitcoinDataPipeline()
    df = pipeline.get_complete_dataset(historical_days=90)
    
    if df is None:
        print("❌ No Bitcoin data available")
        return
    
    print(f"✅ Got {len(df)} days of Bitcoin data")
    
    # Run backtest with problematic parameters
    try:
        results = evaluate_bitcoin_strategy_with_backtest(df, problem_params)
        
        print(f"\n📊 Results with NEGATIVE stop loss:")
        print(f"   Return: {results['returns']:.1%}")
        print(f"   Trades: {results['trades']}")
        print(f"   Win Rate: {results.get('win_rate', 0):.1%}")
        print(f"   Max Drawdown: {results['max_drawdown']:.1%}")
        print(f"   Fitness: {results['fitness']:.4f}")
        
        # Analyze what negative stop loss means
        print(f"\n🔍 ANALYSIS:")
        print(f"   With -2.11% stop loss:")
        print(f"   Exit condition: unrealized_pnl <= -(-0.0211)")
        print(f"   This means: EXIT when profit >= +2.11%")
        print(f"   So negative stop loss = EARLY PROFIT TAKING!")
        
    except Exception as e:
        print(f"❌ Error testing problematic parameters: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with corrected parameters
    print(f"\n🔧 Testing with CORRECTED parameters...")
    
    corrected_params = BitcoinParams(
        buy_threshold_pct=0.279,     # Keep same
        sell_threshold_pct=0.858,    # Keep same
        stop_loss_pct=0.0211,        # FIX: Make positive 2.11%
        take_profit_pct=0.179,       # Keep same
        max_position_pct=1.992       # Keep same
    )
    
    print(f"✅ Corrected Parameters:")
    print(f"   Buy Threshold: {corrected_params.buy_threshold_pct:.1%}")
    print(f"   Sell Threshold: {corrected_params.sell_threshold_pct:.1%}")
    print(f"   Stop Loss: {corrected_params.stop_loss_pct:.2%} ✅ POSITIVE!")
    print(f"   Take Profit: {corrected_params.take_profit_pct:.1%}")
    print(f"   Max Position: {corrected_params.max_position_pct:.1%}")
    
    try:
        corrected_results = evaluate_bitcoin_strategy_with_backtest(df, corrected_params)
        
        print(f"\n📊 Results with CORRECTED stop loss:")
        print(f"   Return: {corrected_results['returns']:.1%}")
        print(f"   Trades: {corrected_results['trades']}")
        print(f"   Win Rate: {corrected_results.get('win_rate', 0):.1%}")
        print(f"   Max Drawdown: {corrected_results['max_drawdown']:.1%}")
        print(f"   Fitness: {corrected_results['fitness']:.4f}")
        
    except Exception as e:
        print(f"❌ Error testing corrected parameters: {e}")

def test_constraint_function():
    """
    Test the constraint function directly
    """
    print(f"\n🔧 TESTING CONSTRAINT FUNCTION:")
    print("=" * 35)
    
    # Simulate an individual with negative stop loss
    test_individual = [
        1.5,    # risk_reward
        1.2,    # trend_strength
        0.8,    # entry_threshold
        1.0,    # confidence
        0.279,  # buy_threshold_pct
        0.858,  # sell_threshold_pct
        1.2,    # bull_multiplier
        0.8,    # bear_multiplier
        0.6,    # high_vol_multiplier
        1.4,    # low_vol_multiplier
        1.99,   # max_position_pct
        -0.0211, # stop_loss_pct (NEGATIVE!)
        0.179   # take_profit_pct
    ]
    
    print(f"🧪 Before constraints:")
    print(f"   Stop Loss: {test_individual[11]:.4f} ({test_individual[11]*100:.2f}%)")
    
    # Apply constraints (mimicking the optimizer function)
    test_individual[11] = max(0.005, min(0.05, test_individual[11]))
    
    print(f"✅ After constraints:")
    print(f"   Stop Loss: {test_individual[11]:.4f} ({test_individual[11]*100:.2f}%)")
    
    if test_individual[11] == 0.005:
        print(f"   ✅ Constraint worked! Negative value clamped to minimum 0.5%")
    else:
        print(f"   ❌ Constraint failed!")

def main():
    investigate_stop_loss_issue()
    test_constraint_function()
    
    print(f"""
🎯 INVESTIGATION SUMMARY:
   1. Negative stop loss likely means EARLY PROFIT TAKING
   2. -2.11% stop = Exit when profit reaches +2.11%
   3. This could explain the excellent performance!
   4. But constraints should prevent negative values
   5. Need to check why constraints aren't being applied properly
   
🔧 POSSIBLE SOLUTIONS:
   1. Fix constraint application in DEAP evolution
   2. Add parameter validation in backtest function
   3. Investigate why negative values slip through
   4. Consider if this "bug" actually discovered a good strategy!""")

if __name__ == "__main__":
    main()
