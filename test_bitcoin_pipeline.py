#!/usr/bin/env python3
"""
Quick test of the fixed Bitcoin data pipeline
"""

import sys
sys.path.append('/home/tkinder/python_projects/avax_evolve')

from bitcoin_data_pipeline import BitcoinDataPipeline

def quick_test():
    print("🧪 Testing Fixed Bitcoin Data Pipeline...")
    
    pipeline = BitcoinDataPipeline()
    df = pipeline.get_complete_dataset(historical_days=90)
    
    if df is not None:
        print(f"\n✅ SUCCESS! Bitcoin pipeline working!")
        print(f"   Dataset: {len(df)} days")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
        print(f"   Ready for DEAP optimization: {'✅' if len(df) >= 60 else '❌'}")
        
        # Quick stats
        returns = df['close'].pct_change().dropna()
        print(f"\n₿ Bitcoin Stats:")
        print(f"   Daily volatility: {returns.std():.3f}")
        print(f"   Max daily gain: {returns.max():.3f}")
        print(f"   Max daily loss: {returns.min():.3f}")
        
        return True
    else:
        print(f"❌ Bitcoin pipeline still failing")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print(f"\n🚀 Ready to run Bitcoin DEAP optimization!")
        print(f"   Run: python bitcoin_optimizer.py")
    else:
        print(f"\n🔧 Pipeline needs more fixes")
