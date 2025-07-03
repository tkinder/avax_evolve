#!/usr/bin/env python3
"""
Quick test to verify enhanced Bitcoin validation works
"""

import sys
sys.path.append('/home/tkinder/python_projects/avax_evolve')

def test_imports():
    print("🧪 Testing Enhanced Bitcoin Imports...")
    
    try:
        from bitcoin_backtester import BitcoinParams, evaluate_bitcoin_strategy_with_backtest
        print("✅ Bitcoin backtester imports working")
        
        from src.evolution.fitness import BacktestResult, calculate_backtest_fitness, FitnessConfig
        print("✅ Fitness system imports working")
        
        from bitcoin_data_pipeline import BitcoinDataPipeline
        print("✅ Bitcoin data pipeline imports working")
        
        print("\n🎉 All enhanced Bitcoin imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def quick_bitcoin_test():
    """Quick test of Bitcoin enhanced validation"""
    try:
        from bitcoin_data_pipeline import BitcoinDataPipeline
        from bitcoin_backtester import BitcoinParams, evaluate_bitcoin_strategy_with_backtest
        
        print("\n📊 Testing Bitcoin data pipeline...")
        pipeline = BitcoinDataPipeline()
        df = pipeline.get_complete_dataset(historical_days=90)
        
        if df is None:
            print("❌ No Bitcoin data available")
            return False
            
        print(f"✅ Got {len(df)} days of Bitcoin data")
        
        print("\n🧪 Testing Bitcoin enhanced backtester...")
        test_params = BitcoinParams(
            buy_threshold_pct=0.70,    # Test conservative entry
            sell_threshold_pct=0.75,   # Test conservative exit
            stop_loss_pct=0.02,        # 2% stop loss
            take_profit_pct=0.05,      # 5% take profit
            max_position_pct=0.8       # 80% position size
        )
        
        results = evaluate_bitcoin_strategy_with_backtest(df, test_params)
        
        print(f"✅ Enhanced Bitcoin Backtester Results:")
        print(f"   Fitness Score: {results['fitness']:.4f}")
        print(f"   Return: {results['returns']:.1%}")
        print(f"   Trades: {results['trades']}")
        print(f"   Max Drawdown: {results['max_drawdown']:.1%}")
        
        print("\n🎉 Enhanced Bitcoin validation working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Bitcoin test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔧 TESTING ENHANCED BITCOIN VALIDATION")
    print("=" * 40)
    
    # Test imports first
    if not test_imports():
        return
    
    # Test functionality
    if not quick_bitcoin_test():
        return
        
    print(f"""
🚀 Enhanced Bitcoin validation ready!
   ✅ All imports working
   ✅ Real backtest simulation working
   ✅ AVAX's proven fitness calculation
   ✅ Ready for enhanced optimization!
   
Next: python bitcoin_enhanced_optimizer.py""")

if __name__ == "__main__":
    main()
