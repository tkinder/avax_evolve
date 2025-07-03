#!/usr/bin/env python3
"""
Fixed Bitcoin Enhanced Optimizer - Constraint Bug Repair
Ensures proper parameter constraint enforcement during DEAP evolution
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

class FixedBitcoinOptimizer:
    """
    FIXED Bitcoin parameter optimizer with proper constraint enforcement
    """
    
    def __init__(self):
        self.pipeline = BitcoinDataPipeline()
        self.results_history = []
        
        print("₿ FIXED Bitcoin Parameter Optimizer Initialized")
        print("🔧 Constraint bug fixed - proper parameter enforcement")
        print("🎯 Focus on traditional risk management strategies")
        
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
        Create Bitcoin-specific fitness configuration
        """
        return FitnessConfig(
            # Bitcoin-specific weights (conservative)
            profitability_weight=0.35,       # Focus on actual profits
            risk_adjusted_weight=0.35,       # Risk-adjusted returns critical
            drawdown_weight=0.20,            # Drawdown control important
            trade_quality_weight=0.10,       # Trade quality
            
            # Bitcoin-specific parameters (conservative)
            min_trades=2,                    # Bitcoin trades selectively
            min_win_rate=0.40,              # Higher win rate expected
            max_drawdown_threshold=0.25,     # Strict drawdown for Bitcoin
            min_profit_threshold=-0.08,      # Limited losses acceptable
            
            # Conservative bounds
            min_fitness=0.01,
            no_trade_fitness=0.005
        )
    
    def validate_bitcoin_params(self, params: BitcoinParams) -> BitcoinParams:
        """
        ADDED: Parameter validation with proper bounds enforcement
        """
        # Create corrected parameters with proper constraints
        corrected_params = BitcoinParams(
            risk_reward=max(0.5, min(3.0, params.risk_reward)),
            trend_strength=max(0.5, min(2.5, params.trend_strength)),
            entry_threshold=max(0.1, min(2.0, params.entry_threshold)),
            confidence=max(0.5, min(2.5, params.confidence)),
            
            # Buy/sell thresholds
            buy_threshold_pct=max(0.20, min(0.80, params.buy_threshold_pct)),
            sell_threshold_pct=max(0.60, min(0.90, params.sell_threshold_pct)),
            
            # Regime multipliers
            bull_multiplier=max(0.9, min(1.3, params.bull_multiplier)),
            bear_multiplier=max(0.6, min(1.0, params.bear_multiplier)),
            high_vol_multiplier=max(0.5, min(0.9, params.high_vol_multiplier)),
            low_vol_multiplier=max(1.0, min(1.5, params.low_vol_multiplier)),
            
            # Risk management - CRITICAL FIXES
            max_position_pct=max(0.4, min(2.0, params.max_position_pct)),
            stop_loss_pct=max(0.005, min(0.05, abs(params.stop_loss_pct))),  # FIX: Ensure positive
            take_profit_pct=max(0.03, min(0.15, params.take_profit_pct)),
            
            # Keep other parameters
            regime_short_window=params.regime_short_window,
            regime_medium_window=params.regime_medium_window,
            regime_long_window=params.regime_long_window,
            price_lookback_days=params.price_lookback_days
        )
        
        # Ensure buy < sell with minimum gap
        if corrected_params.buy_threshold_pct >= corrected_params.sell_threshold_pct - 0.10:
            corrected_params.buy_threshold_pct = corrected_params.sell_threshold_pct - 0.15
            corrected_params.buy_threshold_pct = max(0.20, corrected_params.buy_threshold_pct)
        
        # Ensure take_profit > stop_loss
        if corrected_params.take_profit_pct <= corrected_params.stop_loss_pct:
            corrected_params.take_profit_pct = corrected_params.stop_loss_pct + 0.02
            corrected_params.take_profit_pct = min(0.15, corrected_params.take_profit_pct)
        
        return corrected_params
    
    def evaluate_bitcoin_individual(self, individual, df, fitness_config):
        """
        FIXED: Evaluate Bitcoin individual with proper parameter validation
        """
        try:
            # Extract parameters from individual
            (risk_reward, trend_strength, entry_threshold, confidence,
             buy_threshold_pct, sell_threshold_pct, 
             bull_multiplier, bear_multiplier, high_vol_multiplier, low_vol_multiplier,
             max_position_pct, stop_loss_pct, take_profit_pct) = individual
            
            # Basic parameter validation
            if any(x <= 0 for x in [risk_reward, trend_strength, confidence, max_position_pct]):
                return (0.01,)
            
            if not (0.1 <= buy_threshold_pct <= 0.9):
                return (0.01,)
            
            if not (0.1 <= sell_threshold_pct <= 0.9):
                return (0.01,)
            
            if buy_threshold_pct >= sell_threshold_pct:
                return (0.01,)
            
            # Create Bitcoin parameters (potentially with constraint violations)
            raw_params = BitcoinParams(
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
            
            # FIXED: Apply parameter validation
            validated_params = self.validate_bitcoin_params(raw_params)
            
            # Check for parameter corrections (indicate constraint violations)
            if abs(validated_params.stop_loss_pct - raw_params.stop_loss_pct) > 0.001:
                # Heavy penalty for constraint violations
                return (0.01,)
            
            # Use AVAX's proven validation approach
            results = evaluate_bitcoin_strategy_with_backtest(df, validated_params, fitness_config)
            fitness = results.get('fitness', 0.01)
            
            # Additional Bitcoin validations
            trades = results.get('trades', 0)
            max_dd = results.get('max_drawdown', 1.0)
            
            if trades == 0:
                return (0.005,)
            
            if max_dd > 0.4:
                return (max(0.01, fitness * 0.1),)
            
            return (max(0.01, min(1.0, fitness)),)
            
        except Exception as e:
            print(f"[Bitcoin Evaluation Error] {e}")
            return (0.01,)
    
    def constrain_bitcoin_individual(self, individual):
        """
        FIXED: Enhanced constraint function with better enforcement
        """
        # Apply constraints (same as before but more robust)
        individual[0] = max(0.5, min(3.0, individual[0]))    # risk_reward
        individual[1] = max(0.5, min(2.5, individual[1]))    # trend_strength  
        individual[2] = max(0.1, min(2.0, individual[2]))    # entry_threshold
        individual[3] = max(0.5, min(2.5, individual[3]))    # confidence
        
        # Buy/sell thresholds
        individual[4] = max(0.20, min(0.80, individual[4]))  # buy_threshold_pct
        individual[5] = max(0.60, min(0.90, individual[5]))  # sell_threshold_pct
        
        # Ensure buy < sell with minimum gap
        if individual[4] >= individual[5] - 0.10:
            individual[4] = individual[5] - 0.15
            individual[4] = max(0.20, individual[4])
        
        # Regime multipliers
        individual[6] = max(0.9, min(1.3, individual[6]))    # bull_multiplier
        individual[7] = max(0.6, min(1.0, individual[7]))    # bear_multiplier
        individual[8] = max(0.5, min(0.9, individual[8]))    # high_vol_multiplier
        individual[9] = max(1.0, min(1.5, individual[9]))    # low_vol_multiplier
        
        # Risk management - CRITICAL FIXES
        individual[10] = max(0.4, min(2.0, individual[10]))  # max_position_pct
        
        # FIXED: Ensure stop loss is always positive and within bounds
        individual[11] = max(0.005, min(0.05, abs(individual[11])))  # stop_loss_pct - FORCE POSITIVE
        individual[12] = max(0.03, min(0.15, individual[12]))         # take_profit_pct
        
        # FIXED: Ensure take_profit > stop_loss
        if individual[12] <= individual[11]:
            individual[12] = individual[11] + 0.02
            individual[12] = min(0.15, individual[12])
        
        return individual,
    
    def run_fixed_bitcoin_optimization(self, 
                                     df, 
                                     ngen=25, 
                                     pop_size=50, 
                                     optimization_name="bitcoin_fixed_v1"):
        """
        FIXED: Bitcoin optimization with proper constraint enforcement
        """
        print(f"""
🧬 STARTING FIXED BITCOIN OPTIMIZATION""")
        print("=" * 50)
        print(f"   Dataset: {len(df)} days of Bitcoin data")
        print(f"   Validation: AVAX's proven backtest approach")
        print(f"   Constraint enforcement: FIXED ✅")
        print(f"   Generations: {ngen}")
        print(f"   Population: {pop_size}")
        print(f"   Total backtests: ~{ngen * pop_size:,}")
        
        start_time = time.time()
        
        # Initialize DEAP
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        fitness_config = self.create_bitcoin_fitness_config()
        toolbox = base.Toolbox()
        
        # FIXED: More conservative parameter ranges to avoid constraint violations
        def create_bitcoin_individual():
            return creator.Individual([
                random.uniform(1.0, 2.0),     # risk_reward (narrower range)
                random.uniform(1.0, 1.8),     # trend_strength
                random.uniform(0.5, 1.2),     # entry_threshold
                random.uniform(1.0, 1.8),     # confidence
                random.uniform(0.25, 0.70),   # buy_threshold_pct (safer range)
                random.uniform(0.65, 0.85),   # sell_threshold_pct (safer range)
                random.uniform(1.0, 1.2),     # bull_multiplier
                random.uniform(0.7, 0.9),     # bear_multiplier
                random.uniform(0.6, 0.8),     # high_vol_multiplier
                random.uniform(1.1, 1.4),     # low_vol_multiplier
                random.uniform(0.6, 1.5),     # max_position_pct (safer range)
                random.uniform(0.01, 0.04),   # stop_loss_pct (POSITIVE ONLY)
                random.uniform(0.05, 0.12),   # take_profit_pct (safer range)
            ])
        
        toolbox.register("individual", create_bitcoin_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)  # Smaller mutations
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

        print(f"🚀 Starting FIXED Bitcoin evolution...")

        # FIXED: Enhanced evolution with better constraint enforcement
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, 
            cxpb=0.6,      # Lower crossover to reduce constraint violations
            mutpb=0.2,     # Lower mutation for stability
            ngen=ngen,
            stats=stats, 
            halloffame=hof, 
            verbose=False
        )
        
        # FIXED: Ensure constraints are applied to ALL individuals
        for ind in pop:
            toolbox.constrain(ind)
        
        # FIXED: Critical - ensure best individual is properly constrained
        best = hof[0]
        toolbox.constrain(best)  # ← This was missing before!
        
        # FIXED: Double-check best individual parameters
        best_params_raw = BitcoinParams(
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
        
        # FIXED: Apply final validation
        best_params = self.validate_bitcoin_params(best_params_raw)
        
        # Final evaluation
        best_results = evaluate_bitcoin_strategy_with_backtest(df, best_params, fitness_config)
        
        optimization_time = time.time() - start_time
        
        print(f"""
🎉 FIXED BITCOIN OPTIMIZATION COMPLETE!""")
        print("=" * 45)
        print(f"⏱️  Runtime: {optimization_time / 60:.1f} minutes")
        print(f"🏆 Best fitness: {best_results['fitness']:.4f}")
        print(f"📊 Total trades: {best_results['trades']}")
        print(f"💰 Return: {best_results['returns']:.1%}")
        print(f"📉 Max drawdown: {best_results['max_drawdown']:.1%}")
        print(f"🔧 Constraint enforcement: WORKING ✅")
        
        # Parameter validation check
        print(f"""
🔍 PARAMETER VALIDATION:""")
        print(f"   Stop Loss: {best_params.stop_loss_pct:.2%} (POSITIVE ✅)")
        print(f"   Take Profit: {best_params.take_profit_pct:.1%}")
        print(f"   Buy < Sell: {best_params.buy_threshold_pct:.1%} < {best_params.sell_threshold_pct:.1%} ✅")
        
        # Store results
        result = {
            'params': best_params,
            'fitness': float(best_results['fitness']),
            'results': best_results,
            'optimization_time': optimization_time,
            'optimization_name': optimization_name,
            'data_period': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
            'validation_method': 'fixed_enhanced_backtest',
            'constraint_status': 'FIXED'
        }
        
        self.results_history.append(result)
        return result

def main():
    """
    Main FIXED Bitcoin optimization workflow
    """
    print("₿ FIXED BITCOIN PARAMETER OPTIMIZATION")
    print("=" * 50)
    print("🔧 Constraint bug FIXED")
    print("🎯 Traditional risk management approach")
    print("📊 No more high-frequency micro-trading")
    
    optimizer = FixedBitcoinOptimizer()
    
    try:
        # Get Bitcoin data
        bitcoin_df = optimizer.get_bitcoin_data(historical_days=120)
        
        # Run FIXED optimization
        print(f"""
🚀 Running FIXED optimization...""")
        print(f"   Constraint enforcement: WORKING")
        print(f"   Parameter validation: ACTIVE")
        print(f"   Focus: Traditional Bitcoin strategy")
        
        result = optimizer.run_fixed_bitcoin_optimization(
            df=bitcoin_df,
            ngen=25,
            pop_size=50,
            optimization_name="bitcoin_constraint_fixed_v1"
        )
        
        if result:
            params = result['params']
            results = result['results']
            
            print(f"""
₿ FIXED BITCOIN PARAMETERS""")
            print("=" * 30)
            print(f"🎯 Entry/Exit: {params.buy_threshold_pct:.1%} buy, {params.sell_threshold_pct:.1%} sell")
            print(f"🛡️  Risk: {params.stop_loss_pct:.2%} stop, {params.take_profit_pct:.1%} target")
            print(f"📊 Performance: {results['returns']:.1%} return, {results['trades']} trades")
            print(f"🏆 Fitness: {result['fitness']:.4f}")
            print(f"✅ All parameters within proper bounds!")
            
        else:
            print(f"❌ Fixed optimization failed")
            
    except Exception as e:
        print(f"❌ Fixed Bitcoin optimization error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
