#!/usr/bin/env python3
"""
Solana Parameter Optimization using DEAP
Adapts the successful Bitcoin DEAP framework for Solana-specific optimization
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

# Import Solana data pipeline
from solana_data_pipeline import SolanaDataPipeline

class SolanaParameterOptimizer:
    """
    Solana-specific parameter optimization using DEAP genetic algorithms
    Building on successful Bitcoin optimization results
    """
    
    def __init__(self):
        self.pipeline = SolanaDataPipeline()
        self.results_history = []
        
        print("🌟 Solana Parameter Optimizer Initialized")
        print("🧬 Using proven DEAP genetic algorithm framework")
        print("💡 Building on successful Bitcoin optimization")
        
    def get_solana_data(self, historical_days=120):
        """
        Get Solana data for optimization
        """
        print(f"""
📊 Fetching {historical_days} days of Solana data...""")
        
        df = self.pipeline.get_complete_dataset(historical_days=historical_days)
        
        if df is None or len(df) < 60:
            raise ValueError(f"Insufficient Solana data: {len(df) if df is not None else 0} days")
            
        print(f"✅ Solana data ready: {len(df)} days")
        return df
    
    def create_solana_fitness_config(self):
        """
        Create Solana-specific fitness configuration
        Based on Bitcoin success but adapted for Solana's higher volatility
        """
        return FitnessConfig(
            # Solana-specific weights (balanced approach)
            profitability_weight=0.40,      # Higher focus on profit (Solana can be more aggressive)
            risk_adjusted_weight=0.30,      # Important but not as critical as Bitcoin
            drawdown_weight=0.20,           # Moderate drawdown control
            trade_quality_weight=0.10,      # Trade quality important
            
            # Solana-specific parameters (more aggressive than Bitcoin)
            min_trades=5,                   # Solana might trade more frequently
            min_win_rate=0.30,             # Allow lower win rate for higher volatility
            max_drawdown_threshold=0.40,    # Allow slightly higher drawdown than Bitcoin
            min_profit_threshold=-0.20,     # Allow more losses given higher volatility
            
            # Moderate bounds
            min_fitness=0.01,
            no_trade_fitness=0.005
        )
    
    def run_solana_optimization(self, 
                               df, 
                               ngen=30, 
                               pop_size=60, 
                               optimization_name="solana_v1"):
        """
        Run DEAP optimization specifically for Solana
        """
        print(f"""
🧬 STARTING SOLANA DEAP OPTIMIZATION""")
        print("=" * 50)
        print(f"   Dataset: {len(df)} days of Solana data")
        print(f"   Generations: {ngen}")
        print(f"   Population: {pop_size}")
        print(f"   Total backtests: ~{ngen * pop_size:,}")
        print(f"   Estimated runtime: {(ngen * pop_size) / 40:.0f}-{(ngen * pop_size) / 30:.0f} minutes")
        print(f"   Volatility expectation: Higher than Bitcoin")
        
        start_time = time.time()
        
        # Use Solana-specific fitness configuration
        fitness_config = self.create_solana_fitness_config()
        
        # Create a wrapper function for Solana evaluation
        def solana_evaluate_strategy(df_data, params, fitness_cfg):
            """Solana-specific strategy evaluation"""
            try:
                results = evaluate_adaptive_strategy(df_data, params, fitness_cfg)
                
                # Solana-specific adjustments
                # Reward strategies that can handle Solana's volatility
                volatility_bonus = 1.0
                if results.get('max_drawdown', 1.0) < 0.25:  # Less than 25% drawdown
                    volatility_bonus = 1.15  # 15% bonus for handling volatility well
                
                # Penalize strategies that are too conservative for Solana
                if results.get('trades', 0) < 3:  # Very few trades
                    results['fitness'] *= 0.8  # 20% penalty
                
                # Bonus for good risk-adjusted returns on volatile asset
                if results.get('returns', 0) > 0.05 and results.get('max_drawdown', 1.0) < 0.20:
                    results['fitness'] *= 1.1  # 10% bonus
                
                results['fitness'] *= volatility_bonus
                
                return results
                
            except Exception as e:
                print(f"❌ Solana evaluation error: {e}")
                return {'fitness': 0.01, 'trades': 0, 'returns': -1.0}
        
        # Run DEAP optimization with Solana data
        try:
            result = run_adaptive_optimization(
                df=df,
                ngen=ngen,
                pop_size=pop_size,
                seed=int(time.time()) % 1000000
            )
            
            optimization_time = time.time() - start_time
            
            print(f"""
🎉 SOLANA OPTIMIZATION COMPLETE!""")
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
            print(f"❌ Solana optimization failed: {e}")
            return None
    
    def display_optimized_parameters(self, result):
        """
        Display the optimized Solana parameters in a readable format
        """
        if result is None:
            print("❌ No optimization results to display")
            return
        
        params = result['params']
        
        print(f"""
🌟 OPTIMIZED SOLANA PARAMETERS""")
        print("=" * 35)
        
        print(f"""
🎯 Entry/Exit Thresholds:""")
        print(f"   Buy Threshold:  {params.buy_threshold_pct:.1%} (buy when Solana in bottom {params.buy_threshold_pct:.1%} of range)")
        print(f"   Sell Threshold: {params.sell_threshold_pct:.1%} (sell when Solana in top {(1-params.sell_threshold_pct):.1%} of range)")
        
        print(f"""
🛡️  Risk Management:""")
        print(f"   Stop Loss:      {params.stop_loss_pct:.1%}")
        print(f"   Take Profit:    {params.take_profit_pct:.1%}")
        print(f"   Max Position:   {params.max_position_pct:.1%}")
        
        print(f"""
🧠 Regime Detection:""")
        print(f"   Short Window:   {params.regime_short_window} days")
        print(f"   Medium Window:  {params.regime_medium_window} days")
        print(f"   Long Window:    {params.regime_long_window} days")
        
        print(f"""
📈 Position Sizing:""")
        print(f"   Bull Multiplier:     {params.bull_multiplier:.2f}x")
        print(f"   Bear Multiplier:     {params.bear_multiplier:.2f}x")
        print(f"   High Vol Multiplier: {params.high_vol_multiplier:.2f}x")
        print(f"   Low Vol Multiplier:  {params.low_vol_multiplier:.2f}x")
        
        print(f"""
🏆 Performance Metrics:""")
        results = result['results']
        print(f"   Fitness Score:  {result['fitness']:.4f}")
        print(f"   Total Return:   {results['returns']:.1%}")
        print(f"   Total Trades:   {results['trades']}")
        print(f"   Win Rate:       {results['wins'] / max(results['trades'], 1):.1%}")
        print(f"   Max Drawdown:   {results['max_drawdown']:.1%}")
        
        # Compare to AVAX and Bitcoin parameters
        print(f"""
📊 COMPARISON TO OTHER ASSETS:""")
        print(f"   AVAX Buy Threshold:    16.2% → Solana: {params.buy_threshold_pct:.1%}")
        print(f"   Bitcoin Buy Threshold: 68.8% → Solana: {params.buy_threshold_pct:.1%}")
        print(f"   AVAX Sell Threshold:   69.1% → Solana: {params.sell_threshold_pct:.1%}")
        print(f"   Bitcoin Sell Threshold: 69.7% → Solana: {params.sell_threshold_pct:.1%}")
        print(f"   AVAX Stop Loss:        8.2%  → Solana: {params.stop_loss_pct:.1%}")
        print(f"   Bitcoin Stop Loss:     0.9%  → Solana: {params.stop_loss_pct:.1%}")
        
        # Analyze strategy positioning
        if params.buy_threshold_pct < 0.30:
            strategy_type = "Aggressive (like AVAX)"
        elif params.buy_threshold_pct > 0.60:
            strategy_type = "Conservative (like Bitcoin)"
        else:
            strategy_type = "Balanced (between AVAX & Bitcoin)"
            
        print(f"""
🎯 SOLANA STRATEGY PROFILE:""")
        print(f"   Strategy Type: {strategy_type}")
        print(f"   Entry Aggressiveness: {params.buy_threshold_pct:.1%}")
        print(f"   Risk Profile: {'High' if params.stop_loss_pct > 0.05 else 'Moderate' if params.stop_loss_pct > 0.02 else 'Low'}")
        
    def save_optimization_results(self, result, filename=None):
        """
        Save optimization results to file
        """
        if result is None:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solana_optimization_{timestamp}.json"
        
        try:
            import json
            
            # Convert result to JSON-serializable format
            save_data = {
                'optimization_name': result.get('optimization_name', 'solana_v1'),
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

    def compare_all_assets(self, solana_result):
        """
        Compare Solana optimization results with AVAX and Bitcoin
        """
        if solana_result is None:
            return
            
        print(f"""
📊 COMPLETE MULTI-ASSET PARAMETER COMPARISON""")
        print("=" * 50)
        
        # Extract Solana parameters
        sol_params = solana_result['params']
        sol_results = solana_result['results']
        
        print(f"""
🎯 BUY THRESHOLDS:""")
        print(f"   AVAX:    16.2% (very aggressive entry)")
        print(f"   Bitcoin: 68.8% (very conservative entry)")
        print(f"   Solana:  {sol_params.buy_threshold_pct:.1%} ({'aggressive' if sol_params.buy_threshold_pct < 0.30 else 'conservative' if sol_params.buy_threshold_pct > 0.60 else 'balanced'})")
        
        print(f"""
🎯 SELL THRESHOLDS:""")
        print(f"   AVAX:    69.1%")
        print(f"   Bitcoin: 69.7%") 
        print(f"   Solana:  {sol_params.sell_threshold_pct:.1%}")
        
        print(f"""
🛡️  STOP LOSS STRATEGIES:""")
        print(f"   AVAX:    8.2% (moderate)")
        print(f"   Bitcoin: 0.9% (very tight)")
        print(f"   Solana:  {sol_params.stop_loss_pct:.1%} ({'tight' if sol_params.stop_loss_pct < 0.03 else 'moderate' if sol_params.stop_loss_pct < 0.06 else 'loose'})")
        
        print(f"""
📈 PERFORMANCE COMPARISON:""")
        print(f"   AVAX Return:    7.8% (proven live trading)")
        print(f"   Bitcoin Return: 4.2% (100% win rate)")
        print(f"   Solana Return:  {sol_results['returns']:.1%}")
        
        print(f"""
🏆 STRATEGY PROFILES:""")
        print(f"   AVAX:    Aggressive growth, frequent trading")
        print(f"   Bitcoin: Conservative, selective entries")
        print(f"   Solana:  {self._classify_solana_strategy(sol_params)}")
        
    def _classify_solana_strategy(self, params):
        """Helper to classify Solana strategy"""
        if params.buy_threshold_pct < 0.25:
            return "High-frequency aggressive (AVAX-like)"
        elif params.buy_threshold_pct > 0.65:
            return "Patient conservative (Bitcoin-like)"
        elif 0.30 <= params.buy_threshold_pct <= 0.50:
            return "Balanced momentum (hybrid approach)"
        else:
            return "Moderate contrarian (unique approach)"

def main():
    """
    Main Solana optimization workflow
    """
    print("🌟 SOLANA PARAMETER OPTIMIZATION WITH DEAP")
    print("=" * 50)
    print("Building on successful Bitcoin optimization...")
    
    optimizer = SolanaParameterOptimizer()
    
    try:
        # Step 1: Get Solana data
        print(f"""
📊 Step 1: Fetching Solana data...""")
        solana_df = optimizer.get_solana_data(historical_days=120)
        
        # Step 2: Run optimization
        print(f"""
🚀 Step 2: Starting DEAP optimization...""")
        print(f"   Expected to find Solana's unique strategy profile")
        print(f"   Will discover if Solana is more like AVAX or Bitcoin")
        print(f"   This will take 30-45 minutes")
        print(f"   Time for another coffee! ☕")
        
        result = optimizer.run_solana_optimization(
            df=solana_df,
            ngen=25,        # 25 generations
            pop_size=50,    # 50 individuals = 1,250 backtests
            optimization_name="solana_deap_v1"
        )
        
        if result:
            # Step 3: Display results
            optimizer.display_optimized_parameters(result)
            
            # Step 4: Save results
            optimizer.save_optimization_results(result)
            
            # Step 5: Multi-asset comparison
            optimizer.compare_all_assets(result)
            
            print(f"""
🎯 NEXT STEPS:""")
            print(f"   ✅ Solana parameters optimized!")
            print(f"   📝 Results saved")
            print(f"   🔍 Strategy profile identified")
            print(f"   🚀 Ready to create Solana trading engine")
            print(f"   💰 Deploy with remaining capital allocation")
            print(f"   📊 Complete multi-asset portfolio ready!")
            
        else:
            print(f"❌ Optimization failed")
            
    except Exception as e:
        print(f"❌ Solana optimization error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
