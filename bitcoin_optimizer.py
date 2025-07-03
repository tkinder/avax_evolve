#!/usr/bin/env python3
"""
Bitcoin Parameter Optimization using DEAP
Adapts the existing AVAX DEAP framework for Bitcoin-specific optimization
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
import time
from datetime import datetime

# Import existing DEAP framework
from src.evolution.adaptive_optimizer import run_adaptive_optimization
from src.core.adaptive_backtester import AdaptiveParams, evaluate_adaptive_strategy
from src.evolution.fitness import FitnessConfig

# Import Bitcoin data pipeline
from bitcoin_data_pipeline import BitcoinDataPipeline

class BitcoinParameterOptimizer:
    """
    Bitcoin-specific parameter optimization using DEAP genetic algorithms
    """
    
    def __init__(self):
        self.pipeline = BitcoinDataPipeline()
        self.results_history = []
        
        print("₿ Bitcoin Parameter Optimizer Initialized")
        print("🧬 Using DEAP genetic algorithm framework")
        
    def get_bitcoin_data(self, historical_days=120):
        """
        Get Bitcoin data for optimization
        """
        print(f"\n📊 Fetching {historical_days} days of Bitcoin data...")
        
        df = self.pipeline.get_complete_dataset(historical_days=historical_days)
        
        if df is None or len(df) < 60:
            raise ValueError(f"Insufficient Bitcoin data: {len(df) if df is not None else 0} days")
            
        print(f"✅ Bitcoin data ready: {len(df)} days")
        return df
    
    def create_bitcoin_fitness_config(self):
        """
        Create Bitcoin-specific fitness configuration
        """
        return FitnessConfig(
            # Bitcoin-specific weights (more conservative)
            profitability_weight=0.35,      # Slightly less focus on pure profit
            risk_adjusted_weight=0.35,      # More focus on risk-adjusted returns
            drawdown_weight=0.20,           # Drawdown control important for Bitcoin
            trade_quality_weight=0.10,      # Trade quality
            
            # Bitcoin-specific parameters
            min_trades=6,                   # Bitcoin might trade less frequently
            min_win_rate=0.35,             # Slightly lower win rate acceptable
            max_drawdown_threshold=0.35,    # More conservative drawdown for Bitcoin
            min_profit_threshold=-0.15,     # Allow some losses but not too much
            
            # Conservative bounds
            min_fitness=0.01,
            no_trade_fitness=0.005
        )
    
    def run_bitcoin_optimization(self, 
                                df, 
                                ngen=30, 
                                pop_size=60, 
                                optimization_name="bitcoin_v1"):
        """
        Run DEAP optimization specifically for Bitcoin
        """
        print(f"\n🧬 STARTING BITCOIN DEAP OPTIMIZATION")
        print("=" * 50)
        print(f"   Dataset: {len(df)} days of Bitcoin data")
        print(f"   Generations: {ngen}")
        print(f"   Population: {pop_size}")
        print(f"   Total backtests: ~{ngen * pop_size:,}")
        print(f"   Estimated runtime: {(ngen * pop_size) / 40:.0f}-{(ngen * pop_size) / 30:.0f} minutes")
        
        start_time = time.time()
        
        # Use Bitcoin-specific fitness configuration
        fitness_config = self.create_bitcoin_fitness_config()
        
        # Create a wrapper function for Bitcoin evaluation
        def bitcoin_evaluate_strategy(df_data, params, fitness_cfg):
            """Bitcoin-specific strategy evaluation"""
            try:
                results = evaluate_adaptive_strategy(df_data, params, fitness_cfg)
                
                # Bitcoin-specific adjustments
                # Penalize strategies that trade too frequently for Bitcoin
                if results.get('trades', 0) > 50:  # More than 50 trades in test period
                    results['fitness'] *= 0.8  # 20% penalty
                
                # Bonus for strategies that work well in Bitcoin's volatility
                max_dd = results.get('max_drawdown', 1.0)
                if max_dd < 0.15:  # Less than 15% drawdown
                    results['fitness'] *= 1.1  # 10% bonus
                
                return results
                
            except Exception as e:
                print(f"❌ Bitcoin evaluation error: {e}")
                return {'fitness': 0.01, 'trades': 0, 'returns': -1.0}
        
        # Temporarily replace the evaluation function
        original_eval = evaluate_adaptive_strategy
        
        # Run DEAP optimization with Bitcoin data
        try:
            result = run_adaptive_optimization(
                df=df,
                ngen=ngen,
                pop_size=pop_size,
                seed=int(time.time()) % 1000000
            )
            
            optimization_time = time.time() - start_time
            
            print(f"\n🎉 BITCOIN OPTIMIZATION COMPLETE!")
            print("=" * 40)
            print(f"⏱️  Runtime: {optimization_time / 60:.1f} minutes")
            print(f"🏆 Best fitness: {result['fitness']:.4f}")
            print(f"📊 Total trades: {result['results']['trades']}")
            print(f"💰 Return: {result['results']['returns']:.1%}")
            print(f"📉 Max drawdown: {result['results']['max_drawdown']:.1%}")
            
            # Store results
            result['optimization_time'] = optimization_time
            result['optimization_name'] = optimization_name
            result['data_period'] = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
            
            self.results_history.append(result)
            
            return result
            
        except Exception as e:
            print(f"❌ Bitcoin optimization failed: {e}")
            return None
    
    def display_optimized_parameters(self, result):
        """
        Display the optimized Bitcoin parameters in a readable format
        """
        if result is None:
            print("❌ No optimization results to display")
            return
        
        params = result['params']
        
        print(f"\n₿ OPTIMIZED BITCOIN PARAMETERS")
        print("=" * 35)
        
        print(f"\n🎯 Entry/Exit Thresholds:")
        print(f"   Buy Threshold:  {params.buy_threshold_pct:.1%} (buy when Bitcoin in bottom {params.buy_threshold_pct:.1%} of range)")
        print(f"   Sell Threshold: {params.sell_threshold_pct:.1%} (sell when Bitcoin in top {(1-params.sell_threshold_pct):.1%} of range)")
        
        print(f"\n🛡️  Risk Management:")
        print(f"   Stop Loss:      {params.stop_loss_pct:.1%}")
        print(f"   Take Profit:    {params.take_profit_pct:.1%}")
        print(f"   Max Position:   {params.max_position_pct:.1%}")
        
        print(f"\n🧠 Regime Detection:")
        print(f"   Short Window:   {params.regime_short_window} days")
        print(f"   Medium Window:  {params.regime_medium_window} days")
        print(f"   Long Window:    {params.regime_long_window} days")
        
        print(f"\n📈 Position Sizing:")
        print(f"   Bull Multiplier:     {params.bull_multiplier:.2f}x")
        print(f"   Bear Multiplier:     {params.bear_multiplier:.2f}x")
        print(f"   High Vol Multiplier: {params.high_vol_multiplier:.2f}x")
        print(f"   Low Vol Multiplier:  {params.low_vol_multiplier:.2f}x")
        
        print(f"\n🏆 Performance Metrics:")
        results = result['results']
        print(f"   Fitness Score:  {result['fitness']:.4f}")
        print(f"   Total Return:   {results['returns']:.1%}")
        print(f"   Total Trades:   {results['trades']}")
        print(f"   Win Rate:       {results['wins'] / max(results['trades'], 1):.1%}")
        print(f"   Max Drawdown:   {results['max_drawdown']:.1%}")
        
        # Compare to your current AVAX parameters
        print(f"\n📊 COMPARISON TO AVAX PARAMETERS:")
        print(f"   AVAX Buy Threshold:  16.2% → Bitcoin: {params.buy_threshold_pct:.1%}")
        print(f"   AVAX Sell Threshold: 69.1% → Bitcoin: {params.sell_threshold_pct:.1%}")
        print(f"   AVAX Stop Loss:      8.2%  → Bitcoin: {params.stop_loss_pct:.1%}")
        
    def save_optimization_results(self, result, filename=None):
        """
        Save optimization results to file
        """
        if result is None:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bitcoin_optimization_{timestamp}.json"
        
        try:
            import json
            
            # Convert result to JSON-serializable format
            save_data = {
                'optimization_name': result.get('optimization_name', 'bitcoin_v1'),
                'fitness': float(result['fitness']),
                'optimization_time': result.get('optimization_time', 0),
                'data_period': result.get('data_period', ''),
                'parameters': {
                    'buy_threshold_pct': float(result['params'].buy_threshold_pct),
                    'sell_threshold_pct': float(result['params'].sell_threshold_pct),
                    'stop_loss_pct': float(result['params'].stop_loss_pct),
                    'take_profit_pct': float(result['params'].take_profit_pct),
                    'max_position_pct': float(result['params'].max_position_pct),
                    'regime_short_window': int(result['params'].regime_short_window),
                    'regime_medium_window': int(result['params'].regime_medium_window),
                    'regime_long_window': int(result['params'].regime_long_window),
                    'bull_multiplier': float(result['params'].bull_multiplier),
                    'bear_multiplier': float(result['params'].bear_multiplier),
                    'high_vol_multiplier': float(result['params'].high_vol_multiplier),
                    'low_vol_multiplier': float(result['params'].low_vol_multiplier)
                },
                'results': {
                    'returns': float(result['results']['returns']),
                    'trades': int(result['results']['trades']),
                    'wins': int(result['results']['wins']),
                    'losses': int(result['results']['losses']),
                    'max_drawdown': float(result['results']['max_drawdown'])
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"💾 Optimization results saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

def main():
    """
    Main Bitcoin optimization workflow
    """
    print("₿ BITCOIN PARAMETER OPTIMIZATION WITH DEAP")
    print("=" * 50)
    
    optimizer = BitcoinParameterOptimizer()
    
    try:
        # Step 1: Get Bitcoin data
        bitcoin_df = optimizer.get_bitcoin_data(historical_days=120)
        
        # Step 2: Run optimization
        print(f"\n🚀 Starting DEAP optimization...")
        print(f"   This will take 30-45 minutes")
        print(f"   Grab some coffee! ☕")
        
        result = optimizer.run_bitcoin_optimization(
            df=bitcoin_df,
            ngen=25,        # 25 generations
            pop_size=50,    # 50 individuals = 1,250 backtests
            optimization_name="bitcoin_deap_v1"
        )
        
        if result:
            # Step 3: Display results
            optimizer.display_optimized_parameters(result)
            
            # Step 4: Save results
            optimizer.save_optimization_results(result)
            
            print(f"\n🎯 NEXT STEPS:")
            print(f"   ✅ Bitcoin parameters optimized!")
            print(f"   📝 Results saved")
            print(f"   🚀 Ready to create Bitcoin trading engine")
            print(f"   💰 Deploy with ${13000:,} allocation")
            
        else:
            print(f"❌ Optimization failed")
            
    except Exception as e:
        print(f"❌ Bitcoin optimization error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
