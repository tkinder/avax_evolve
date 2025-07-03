#!/usr/bin/env python3
"""
Enhanced Bitcoin Parameter Optimizer using AVAX's proven validation approach
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
from bitcoin_backtester import BitcoinParams, evaluate_bitcoin_strategy_with_backtest
from src.evolution.fitness import FitnessConfig

# Import Bitcoin data pipeline
from bitcoin_data_pipeline import BitcoinDataPipeline

# Import DEAP for parameter space exploration
from deap import base, creator, tools, algorithms
import random

class EnhancedBitcoinOptimizer:
    """
    Enhanced Bitcoin parameter optimizer using AVAX's proven backtest validation
    """
    
    def __init__(self):
        self.pipeline = BitcoinDataPipeline()
        self.results_history = []
        
        print("₿ Enhanced Bitcoin Parameter Optimizer Initialized")
        print("🎯 Using AVAX's proven backtest validation approach")
        print("🧬 Real trade execution simulation with slippage & fees")
        
    def get_bitcoin_data(self, historical_days=120):
        """
        Get Bitcoin data for optimization
        """
        print(f"""
📊 Fetching {historical_days} days of Bitcoin data...""")
        
        df = self.pipeline.get_complete_dataset(historical_days=historical_days)
        
        if df is None or len(df) < 60:
            raise ValueError(f"Insufficient Bitcoin data: {len(df) if df is not None else 0} days")
            
        print(f"✅ Bitcoin data ready: {len(df)} days")
        return df
    
    def create_bitcoin_fitness_config(self):
        """
        Create Bitcoin-specific fitness configuration using AVAX's approach
        """
        return FitnessConfig(
            # Bitcoin-specific weights (conservative like AVAX)
            profitability_weight=0.35,       # Focus on actual profits
            risk_adjusted_weight=0.35,       # Risk-adjusted returns critical
            drawdown_weight=0.20,            # Drawdown control important
            trade_quality_weight=0.10,       # Trade quality
            
            # Bitcoin-specific parameters (conservative)
            min_trades=2,                    # Bitcoin trades very selectively
            min_win_rate=0.40,              # Higher win rate expected
            max_drawdown_threshold=0.25,     # Very strict drawdown for Bitcoin
            min_profit_threshold=-0.08,      # Limited losses acceptable
            
            # Conservative bounds
            min_fitness=0.01,
            no_trade_fitness=0.005
        )
    
    def evaluate_bitcoin_individual(self, individual, df, fitness_config):
        """
        Evaluate Bitcoin individual using AVAX's backtest validation approach
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
            
            # Create Bitcoin parameters
            bitcoin_params = BitcoinParams(
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
            results = evaluate_bitcoin_strategy_with_backtest(df, bitcoin_params, fitness_config)
            fitness = results.get('fitness', 0.01)
            
            # Additional Bitcoin validations
            trades = results.get('trades', 0)
            max_dd = results.get('max_drawdown', 1.0)
            
            if trades == 0:
                return (0.005,)
            
            if max_dd > 0.4:  # Conservative drawdown for Bitcoin
                return (max(0.01, fitness * 0.1),)
            
            return (max(0.01, min(1.0, fitness)),)
            
        except Exception as e:
            print(f"[Bitcoin Evaluation Error] {e}")
            return (0.01,)
    
    def constrain_bitcoin_individual(self, individual):
        """
        Ensure Bitcoin parameters stay within reasonable bounds
        """
        # Parameters: risk_reward, trend_strength, entry_threshold, confidence,
        #            buy_threshold_pct, sell_threshold_pct, 
        #            bull_multiplier, bear_multiplier, high_vol_multiplier, low_vol_multiplier,
        #            max_position_pct, stop_loss_pct, take_profit_pct
        
        individual[0] = max(0.5, min(3.0, individual[0]))    # risk_reward
        individual[1] = max(0.5, min(2.5, individual[1]))    # trend_strength  
        individual[2] = max(0.1, min(2.0, individual[2]))    # entry_threshold
        individual[3] = max(0.5, min(2.5, individual[3]))    # confidence
        
        # Buy/sell thresholds - Bitcoin tends toward high buy thresholds
        individual[4] = max(0.20, min(0.80, individual[4]))  # buy_threshold_pct (20-80%)
        individual[5] = max(0.60, min(0.90, individual[5]))  # sell_threshold_pct (60-90%)
        
        # Ensure buy < sell with minimum gap
        if individual[4] >= individual[5] - 0.10:
            individual[4] = individual[5] - 0.15
            individual[4] = max(0.20, individual[4])
        
        # Regime multipliers (conservative for Bitcoin)
        individual[6] = max(0.9, min(1.3, individual[6]))    # bull_multiplier
        individual[7] = max(0.6, min(1.0, individual[7]))    # bear_multiplier
        individual[8] = max(0.5, min(0.9, individual[8]))    # high_vol_multiplier
        individual[9] = max(1.0, min(1.5, individual[9]))    # low_vol_multiplier
        
        # Risk management (Bitcoin-specific bounds)
        individual[10] = max(0.4, min(2.0, individual[10]))  # max_position_pct (40-200%)
        individual[11] = max(0.005, min(0.05, individual[11])) # stop_loss_pct (0.5-5%)
        individual[12] = max(0.03, min(0.15, individual[12])) # take_profit_pct (3-15%)
        
        # Ensure take_profit > stop_loss
        if individual[12] <= individual[11]:
            individual[12] = individual[11] + 0.02
            individual[12] = min(0.15, individual[12])
        
        return individual,
    
    def run_enhanced_bitcoin_optimization(self, 
                                        df, 
                                        ngen=25, 
                                        pop_size=50, 
                                        optimization_name="bitcoin_enhanced_v1"):
        """
        Run enhanced Bitcoin optimization using AVAX's validation approach
        """
        print(f"""
🧬 STARTING ENHANCED BITCOIN OPTIMIZATION""")
        print("=" * 50)
        print(f"   Dataset: {len(df)} days of Bitcoin data")
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
        
        # Use Bitcoin-specific fitness configuration
        fitness_config = self.create_bitcoin_fitness_config()
        
        toolbox = base.Toolbox()
        
        # Bitcoin-specific parameter creation (biased toward discovered optima)
        def create_bitcoin_individual():
            return creator.Individual([
                random.uniform(0.8, 2.5),     # risk_reward
                random.uniform(0.8, 2.0),     # trend_strength
                random.uniform(0.3, 1.5),     # entry_threshold
                random.uniform(0.8, 2.0),     # confidence
                random.uniform(0.30, 0.80),   # buy_threshold_pct (Bitcoin range)
                random.uniform(0.60, 0.85),   # sell_threshold_pct (Bitcoin range)
                random.uniform(1.0, 1.3),     # bull_multiplier
                random.uniform(0.6, 0.8),     # bear_multiplier
                random.uniform(0.5, 0.7),     # high_vol_multiplier
                random.uniform(1.2, 1.5),     # low_vol_multiplier
                random.uniform(0.8, 1.8),     # max_position_pct (Bitcoin allows higher)
                random.uniform(0.005, 0.02),  # stop_loss_pct (Bitcoin tight stops)
                random.uniform(0.04, 0.10),   # take_profit_pct (Bitcoin moderate targets)
            ])
        
        toolbox.register("individual", create_bitcoin_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Genetic operators
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.25)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("constrain", self.constrain_bitcoin_individual)

        def evaluate(ind):
            return self.evaluate_bitcoin_individual(ind, df, fitness_config)

        toolbox.register("evaluate", evaluate)

        # Create population
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        print(f"🚀 Starting Bitcoin evolution with AVAX validation...")

        # Run evolution
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, 
            cxpb=0.7,
            mutpb=0.25,
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
        
        # Convert to BitcoinParams
        best_params = BitcoinParams(
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
        best_results = evaluate_bitcoin_strategy_with_backtest(df, best_params, fitness_config)
        
        optimization_time = time.time() - start_time
        
        print(f"""
🎉 ENHANCED BITCOIN OPTIMIZATION COMPLETE!""")
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
        Display enhanced Bitcoin optimization results
        """
        if result is None:
            print("❌ No optimization results to display")
            return
        
        params = result['params']
        
        print(f"""
₿ ENHANCED BITCOIN PARAMETERS (AVAX Validation)""")
        print("=" * 45)
        
        print(f"""
🎯 Entry/Exit Thresholds:""")
        print(f"   Buy Threshold:  {params.buy_threshold_pct:.1%}")
        print(f"   Sell Threshold: {params.sell_threshold_pct:.1%}")
        
        print(f"""
🛡️  Risk Management:""")
        print(f"   Stop Loss:      {params.stop_loss_pct:.2%}")
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
        print(f"   Previous Bitcoin Buy:  68.8% → Enhanced: {params.buy_threshold_pct:.1%}")
        print(f"   Previous Bitcoin Sell: 69.7% → Enhanced: {params.sell_threshold_pct:.1%}")
        print(f"   Previous Bitcoin Stop: 0.9%  → Enhanced: {params.stop_loss_pct:.2%}")

def main():
    """
    Main enhanced Bitcoin optimization workflow
    """
    print("₿ ENHANCED BITCOIN PARAMETER OPTIMIZATION")
    print("=" * 50)
    print("🎯 Using AVAX's proven validation approach")
    print("🧬 Real backtest validation with trade execution")
    
    optimizer = EnhancedBitcoinOptimizer()
    
    try:
        # Step 1: Get Bitcoin data
        print(f"""
📊 Step 1: Fetching Bitcoin data...""")
        bitcoin_df = optimizer.get_bitcoin_data(historical_days=120)
        
        # Step 2: Run enhanced optimization
        print(f"""
🚀 Step 2: Starting enhanced DEAP optimization...""")
        print(f"   Each individual tested with real backtest simulation")
        print(f"   Trade execution with slippage & fees modeled")
        print(f"   AVAX's proven fitness calculation approach")
        print(f"   This will take 30-45 minutes")
        print(f"   Enhanced validation in progress... ☕")
        
        result = optimizer.run_enhanced_bitcoin_optimization(
            df=bitcoin_df,
            ngen=25,        # 25 generations
            pop_size=50,    # 50 individuals = 1,250 real backtests
            optimization_name="bitcoin_enhanced_v1"
        )
        
        if result:
            # Step 3: Display enhanced results
            optimizer.display_enhanced_results(result)
            
            print(f"""
🎯 ENHANCED OPTIMIZATION COMPLETE:""")
            print(f"   ✅ Bitcoin parameters optimized with AVAX validation!")
            print(f"   📊 Real backtest validation (not synthetic)")
            print(f"   🎯 Trade execution simulation included")
            print(f"   🚀 Ready for deployment with proven validation")
            
        else:
            print(f"❌ Enhanced optimization failed")
            
    except Exception as e:
        print(f"❌ Enhanced Bitcoin optimization error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
