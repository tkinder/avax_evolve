#!/usr/bin/env python3
"""
Quick test of the Solana data pipeline
"""

import sys
sys.path.append('/home/tkinder/python_projects/avax_evolve')

from solana_data_pipeline import SolanaDataPipeline

def quick_test():
    print("🧪 Testing Solana Data Pipeline...")
    
    pipeline = SolanaDataPipeline()
    df = pipeline.get_complete_dataset(historical_days=90)
    
    if df is not None:
        print(f"""
✅ SUCCESS! Solana pipeline working!""")
        print(f"   Dataset: {len(df)} days")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
        print(f"   Ready for DEAP optimization: {'✅' if len(df) >= 60 else '❌'}")
        
        # Quick stats
        returns = df['close'].pct_change().dropna()
        bitcoin_volatility = 0.022  # From previous results
        
        print(f"""
🌟 Solana vs Bitcoin Stats:""")
        print(f"   Solana daily volatility: {returns.std():.3f}")
        print(f"   Bitcoin daily volatility: {bitcoin_volatility:.3f}")
        print(f"   Volatility ratio: {returns.std()/bitcoin_volatility:.1f}x")
        print(f"   Solana max daily gain: {returns.max():.3f}")
        print(f"   Solana max daily loss: {returns.min():.3f}")
        
        # Predict strategy type
        if returns.std() > 0.04:
            predicted_strategy = "Likely aggressive (high volatility)"
        elif returns.std() < 0.025:
            predicted_strategy = "Likely conservative (low volatility)"
        else:
            predicted_strategy = "Likely balanced (moderate volatility)"
            
        print(f"   Predicted strategy: {predicted_strategy}")
        
        return True
    else:
        print(f"❌ Solana pipeline failing")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print(f"""
🚀 Ready to run Solana DEAP optimization!""")
        print(f"   Test first: python test_solana_pipeline.py")
        print(f"   Then run: python solana_optimizer.py")
        print(f"   Expected: Discover Solana's unique strategy profile")
    else:
        print(f"""
🔧 Pipeline needs fixes""")
