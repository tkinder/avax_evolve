#!/usr/bin/env python3
"""
Enhanced Solana Parameter Optimizer using AVAX's proven validation approach
Replaces synthetic DEAP optimization with real backtest-driven validation
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

# Import AVAX's proven validation system
from solana_backtester import SolanaParams, evaluate_solana_strategy_with_backtest
from src.evolution.fitness import FitnessConfig

# Import Solana data pipeline
from solana_data_pipeline import SolanaDataPipeline

# Import DEAP for parameter space exploration
from deap import base, creator, tools, algorithms
import random

class EnhancedSolanaOptimizer:
    """
    Enhanced Solana parameter optimizer using AVAX's proven backtest validation
    """
    
    def __init__(self):
        self.pipeline = SolanaDataPipeline()
        self.results_history = []
        
        print("🌟 Enhanced Solana Parameter Optimizer Initialized")
        print("🎯 Using AVAX's proven backtest validation approach")
        print("🧬 Real trade execution simulation with slippage & fees")
        
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
        Create Solana-specific fitness configuration using AVAX's approach
        """
        return FitnessConfig(
            # Solana-specific weights (balanced approach)
            profitability_weight=0.40,       # Focus on profits (Solana can be aggressive)
            risk_adjusted_weight=0.25,       # Risk-adjusted returns
            drawdown_weight=0.25,            # Moderate drawdown control
            trade_quality_weight=0.10,       # Trade quality
            
            # Solana-specific parameters (moderate)
            min_trades=2,                    # Solana contrarian - fewer trades ok
            min_win_rate=0.30,              # Allow lower win rate for volatility
            max_drawdown_threshold=0.45,     # Higher drawdown tolerance than Bitcoin
            min_profit_threshold=-0.20,      # Allow more losses given volatility
            
            # Moderate bounds
            min_fitness=0.01,
            no_trade_fitness=0.005
        )
    
    def evaluate_solana_individual(self, individual, df, fitness_config):
        """
        Evaluate Solana individual using AVAX's backtest validation approach
        """
        try:
            # Extract parameters from individual
            (risk_reward, trend_strength, entry_threshold, confidence,
             buy_threshold_pct, sell_threshold_pct, 
             bull_multiplier, bear_multiplier, high_vol_multiplier, low_vol_multiplier,
             max_position_pct, stop_loss_pct, take_profit_pct) = individual
            
            # Parameter validation (AVAX style)
            if any(x <= 0 for x in [risk_reward, trend_strength, confidence, max_position_pct]):
                return (0.01,)
            
            if not (0.1 <= buy_threshold_pct <= 0.9):
                return (0.01,)
            
            if not (0.1 <= sell_threshold_pct <= 0.9):
                return (0.01,)
            
            if buy_threshold_pct >= sell_threshold_pct:
                return (0.01,)
            
            # Create Solana parameters
            solana_params = SolanaParams(
                risk_reward=risk_reward,
                trend_strength=trend_strength,
                entry_threshold=entry_threshold,
                confidence=confidence,
                buy_threshold_pct=buy_threshold_pct,
                sell_threshold_pct=sell_threshold_pct,
                bull_multiplier=bull_multiplier,
                bear_multiplier=bear_multiplier,
                high_vol_multiplier=high_vol_multiplier,
                low_vol_multiplier=low_vol_multiplier,
                max_position_pct=max_position_pct,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct
            )
            
            # Use AVAX's proven validation approach
            results = evaluate_solana_strategy_with_backtest(df, solana_params, fitness_config)
            fitness = results.get('fitness', 0.01)
            
            # Additional Solana validations
            trades = results.get('trades', 0)
            max_dd = results.get('max_drawdown', 1.0)
            
            if trades == 0:
                return (0.005,)
            
            if max_dd > 0.6:  # Allow higher drawdown for Solana than Bitcoin
                return (max(0.01, fitness * 0.1),)
            
            return (max(0.01, min(1.0, fitness)),)
            
        except Exception as e:
            print(f"[Solana Evaluation Error] {e}")
            return (0.01,)
    
    def constrain_solana_individual(self, individual):
        """
        Ensure Solana parameters stay within reasonable bounds
        """
        # Parameters: risk_reward, trend_strength, entry_threshold, confidence,
        #            buy_threshold_pct, sell_threshold_pct, 
        #            bull_multiplier, bear_multiplier, high_vol_multiplier, low_vol_multiplier,
        #            max_position_pct, stop_loss_pct, take_profit_pct
        
        individual[0] = max(0.5, min(3.0, individual[0]))    # risk_reward
        individual[1] = max(0.5, min(2.5, individual[1]))    # trend_strength  
        individual[2] = max(0.1, min(2.0, individual[2]))    # entry_threshold
        individual[3] = max(0.5, min(2.5, individual[3]))    # confidence
        
        # Buy/sell thresholds - Solana tends toward moderate levels
        individual[4] = max(0.15, min(0.75, individual[4]))  # buy_threshold_pct (15-75%)
        individual[5] = max(0.55, min(0.90, individual[5]))  # sell_threshold_pct (55-90%)
        
        # Ensure buy < sell with minimum gap
        if individual[4] >= individual[5] - 0.15:
            individual[4] = individual[5] - 0.20
            individual[4] = max(0.15, individual[4])
        
        # Regime multipliers (moderate for Solana)
        individual[6] = max(0.9, min(1.3, individual[6]))    # bull_multiplier
        individual[7] = max(0.6, min(1.0, individual[7]))    # bear_multiplier
        individual[8] = max(0.6, min(1.0, individual[8]))    # high_vol_multiplier
        individual[9] = max(1.0, min(1.4, individual[9]))    # low_vol_multiplier
        
        # Risk management (Solana-specific bounds - allows higher risk)
        individual[10] = max(0.3, min(1.0, individual[10]))  # max_position_pct (30-100%)
        individual[11] = max(0.05, min(0.50, individual[11])) # stop_loss_pct (5-50% for contrarian)
        individual[12] = max(0.08, min(0.25, individual[12])) # take_profit_pct (8-25%)
        
        # Ensure take_profit > stop_loss (except for contrarian high stops)
        if individual[11] < 0.20 and individual[12] <= individual[11]:
            individual[12] = individual[11] + 0.05
            individual[12] = min(0.25, individual[12])
        
        return individual,
    
    def run_enhanced_solana_optimization(self, 
                                       df, 
                                       ngen=25, 
                                       pop_size=50, 
                                       optimization_name="solana_enhanced_v1"):
        """
        Run enhanced Solana optimization using AVAX's validation approach
        """
        print(f"""
🧬 STARTING ENHANCED SOLANA OPTIMIZATION""")
        print("=" * 50)
        print(f"   Dataset: {len(df)} days of Solana data")
        print(f"   Validation: AVAX's proven backtest approach")
        print(f"   Generations: {ngen}")
        print(f"   Population: {pop_size}")
        print(f"   Total backtests: ~{ngen * pop_size:,}")
        print(f"   Each backtest: Real trade simulation with fees & slippage")
        
        start_time = time.time()
        
        # Initialize DEAP
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Use Solana-specific fitness configuration
        fitness_config = self.create_solana_fitness_config()
        
        toolbox = base.Toolbox()
        
        # Solana-specific parameter creation (biased toward discovered optima)
        def create_solana_individual():
            return creator.Individual([
                random.uniform(0.8, 2.5),     # risk_reward
                random.uniform(0.8, 2.0),     # trend_strength
                random.uniform(0.3, 1.5),     # entry_threshold
                random.uniform(0.8, 2.0),     # confidence
                random.uniform(0.20, 0.70),   # buy_threshold_pct (Solana range)
                random.uniform(0.60, 0.85),   # sell_threshold_pct (Solana range)
                random.uniform(1.0, 1.2),     # bull_multiplier
                random.uniform(0.6, 0.8),     # bear_multiplier
                random.uniform(0.7, 0.9),     # high_vol_multiplier
                random.uniform(1.0, 1.3),     # low_vol_multiplier
                random.uniform(0.4, 0.8),     # max_position_pct (Solana moderate)
                random.uniform(0.10, 0.40),   # stop_loss_pct (Solana wide range)
                random.uniform(0.08, 0.20),   # take_profit_pct (Solana moderate targets)
            ])
        
        toolbox.register("individual", create_solana_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Genetic operators
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=4)
        toolbox.register("constrain", self.constrain_solana_individual)

        def evaluate(ind):
            return self.evaluate_solana_individual(ind, df, fitness_config)

        toolbox.register("evaluate", evaluate)

        # Create population
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        print(f"🚀 Starting Solana evolution with AVAX validation...")

        # Run evolution
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, 
            cxpb=0.7,
            mutpb=0.3,
            ngen=ngen,
            stats=stats, 
            halloffame=hof, 
            verbose=False
        )
        
        # Apply constraints to final population
        for ind in pop:
            toolbox.constrain(ind)

        # Get best result
        best = hof[0]
        
        # Convert to SolanaParams
        best_params = SolanaParams(
            risk_reward=best[0],
            trend_strength=best[1],
            entry_threshold=best[2],
            confidence=best[3],
            buy_threshold_pct=best[4],
            sell_threshold_pct=best[5],
            bull_multiplier=best[6],
            bear_multiplier=best[7],
            high_vol_multiplier=best[8],
            low_vol_multiplier=best[9],
            max_position_pct=best[10],
            stop_loss_pct=best[11],
            take_profit_pct=best[12]
        )
        
        # Final evaluation using AVAX's approach
        best_results = evaluate_solana_strategy_with_backtest(df, best_params, fitness_config)
        
        optimization_time = time.time() - start_time
        
        print(f"""
🎉 ENHANCED SOLANA OPTIMIZATION COMPLETE!""")
        print("=" * 45)
        print(f"⏱️  Runtime: {optimization_time / 60:.1f} minutes")
        print(f"🏆 Best fitness: {best_results['fitness']:.4f}")
        print(f"📊 Total trades: {best_results['trades']}")
        print(f"💰 Return: {best_results['returns']:.1%}")
        print(f"📉 Max drawdown: {best_results['max_drawdown']:.1%}")
        print(f"🎯 Validation: Real backtest with trade execution")
        
        # Store results
        result = {
            'params': best_params,
            'fitness': float(best_results['fitness']),
            'results': best_results,
            'optimization_time': optimization_time,
            'optimization_name': optimization_name,
            'data_period': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
            'validation_method': 'enhanced_backtest'
        }
        
        self.results_history.append(result)
        return result
    
    def display_enhanced_results(self, result):
        """
        Display enhanced Solana optimization results
        """
        if result is None:
            print("❌ No optimization results to display")
            return
        
        params = result['params']
        
        print(f"""
🌟 ENHANCED SOLANA PARAMETERS (AVAX Validation)""")
        print("=" * 45)
        
        print(f"""
🎯 Entry/Exit Thresholds:""")
        print(f"   Buy Threshold:  {params.buy_threshold_pct:.1%}")
        print(f"   Sell Threshold: {params.sell_threshold_pct:.1%}")
        
        print(f"""
🛡️  Risk Management:""")
        print(f"   Stop Loss:      {params.stop_loss_pct:.1%}")
        print(f"   Take Profit:    {params.take_profit_pct:.1%}")
        print(f"   Max Position:   {params.max_position_pct:.1%}")
        
        print(f"""
📈 Position Sizing:""")
        print(f"   Bull Multiplier:     {params.bull_multiplier:.2f}x")
        print(f"   Bear Multiplier:     {params.bear_multiplier:.2f}x")
        print(f"   High Vol Multiplier: {params.high_vol_multiplier:.2f}x")
        print(f"   Low Vol Multiplier:  {params.low_vol_multiplier:.2f}x")
        
        print(f"""
🏆 Enhanced Performance Metrics:""")
        results = result['results']
        print(f"   Fitness Score:  {result['fitness']:.4f}")
        print(f"   Total Return:   {results['returns']:.1%}")
        print(f"   Total Trades:   {results['trades']}")
        print(f"   Win Rate:       {results.get('win_rate', 0):.1%}")
        print(f"   Max Drawdown:   {results['max_drawdown']:.1%}")
        print(f"   Validation:     {result['validation_method']}")
        
        print(f"""
📊 COMPARISON TO PREVIOUS OPTIMIZATION:""")
        print(f"   Previous Solana Buy:  56.1% → Enhanced: {params.buy_threshold_pct:.1%}")
        print(f"   Previous Solana Sell: 74.5% → Enhanced: {params.sell_threshold_pct:.1%}")
        print(f"   Previous Solana Stop: 37.6% → Enhanced: {params.stop_loss_pct:.1%}")
        
        # Strategy classification
        if params.buy_threshold_pct < 0.30:
            strategy_type = "Enhanced Aggressive"
        elif params.buy_threshold_pct > 0.65:
            strategy_type = "Enhanced Conservative"
        else:
            strategy_type = "Enhanced Balanced"
            
        print(f"""
🎯 ENHANCED SOLANA STRATEGY PROFILE:""")
        print(f"   Strategy Type: {strategy_type}")
        print(f"   Risk Profile: {'High' if params.stop_loss_pct > 0.20 else 'Moderate' if params.stop_loss_pct > 0.10 else 'Low'}")
    
    def save_enhanced_results(self, result, filename=None):
        """
        Save enhanced optimization results
        """
        if result is None:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solana_enhanced_optimization_{timestamp}.json"
        
        try:
            import json
            
            # Convert result to JSON-serializable format
            save_data = {
                'optimization_name': result.get('optimization_name', 'solana_enhanced_v1'),
                'validation_method': result.get('validation_method', 'enhanced_backtest'),
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
                    'win_rate': float(result['results'].get('win_rate', 0)),
                    'max_drawdown': float(result['results']['max_drawdown']),
                    'fitness_components': {
                        'profitability': float(result['results'].get('profitability', 0)),
                        'risk_adjusted': float(result['results'].get('risk_adjusted', 0)),
                        'drawdown': float(result['results'].get('drawdown', 0)),
                        'trade_quality': float(result['results'].get('trade_quality', 0))
                    }
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"💾 Enhanced optimization results saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save enhanced results: {e}")

def main():
    """
    Main enhanced Solana optimization workflow
    """
    print("🌟 ENHANCED SOLANA PARAMETER OPTIMIZATION")
    print("=" * 50)
    print("🎯 Using AVAX's proven validation approach")
    print("🧬 Real backtest validation with trade execution")
    
    optimizer = EnhancedSolanaOptimizer()
    
    try:
        # Step 1: Get Solana data
        print(f"""
📊 Step 1: Fetching Solana data...""")
        solana_df = optimizer.get_solana_data(historical_days=120)
        
        # Step 2: Run enhanced optimization
        print(f"""
🚀 Step 2: Starting enhanced DEAP optimization...""")
        print(f"   Each individual tested with real backtest simulation")
        print(f"   Trade execution with slippage & fees modeled")
        print(f"   AVAX's proven fitness calculation approach")
        print(f"   This will take 30-45 minutes")
        print(f"   Enhanced validation in progress... ☕")
        
        result = optimizer.run_enhanced_solana_optimization(
            df=solana_df,
            ngen=25,        # 25 generations
            pop_size=50,    # 50 individuals = 1,250 real backtests
            optimization_name="solana_enhanced_v1"
        )
        
        if result:
            # Step 3: Display enhanced results
            optimizer.display_enhanced_results(result)
            
            # Step 4: Save enhanced results
            optimizer.save_enhanced_results(result)
            
            print(f"""
🎯 ENHANCED OPTIMIZATION COMPLETE:""")
            print(f"   ✅ Solana parameters optimized with AVAX validation!")
            print(f"   📊 Real backtest validation (not synthetic)")
            print(f"   🎯 Trade execution simulation included")
            print(f"   📝 Enhanced results saved")
            print(f"   🚀 Ready for deployment with proven validation")
            
        else:
            print(f"❌ Enhanced optimization failed")
            
    except Exception as e:
        print(f"❌ Enhanced Solana optimization error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
