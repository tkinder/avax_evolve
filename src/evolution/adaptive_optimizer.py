# src/evolution/adaptive_optimizer.py

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from deap import base, creator, tools, algorithms
import random
import numpy as np
import time
from src.core.adaptive_backtester import AdaptiveParams, evaluate_adaptive_strategy
from src.evolution.fitness import FitnessConfig

# Ensure DEAP classes exist
def _init_deap():
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

_init_deap()

def evaluate_adaptive_individual(individual, df, fitness_config):
    """
    Evaluate an individual using the adaptive strategy.
    """
    try:
        # Convert individual to AdaptiveParams
        (risk_reward, trend_strength, entry_threshold, confidence,
         buy_threshold_pct, sell_threshold_pct, 
         bull_multiplier, bear_multiplier, high_vol_multiplier, low_vol_multiplier,
         max_position_pct, stop_loss_pct, take_profit_pct) = individual
        
        # Validate parameters
        if any(x <= 0 for x in [risk_reward, trend_strength, confidence, max_position_pct]):
            return (0.01,)
        
        if not (0.1 <= buy_threshold_pct <= 0.9):
            return (0.01,)
        
        if not (0.1 <= sell_threshold_pct <= 0.9):
            return (0.01,)
        
        if buy_threshold_pct >= sell_threshold_pct:
            return (0.01,)
        
        # Create adaptive parameters
        params = AdaptiveParams(
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
        
        # Evaluate strategy
        results = evaluate_adaptive_strategy(df, params, fitness_config)
        fitness = results.get('fitness', 0.01)
        
        # Additional validation
        trades = results.get('trades', 0)
        max_dd = results.get('max_drawdown', 1.0)
        
        if trades == 0:
            return (0.005,)
        
        if max_dd > 0.6:  # More than 60% drawdown
            return (max(0.01, fitness * 0.1),)
        
        return (max(0.01, min(1.0, fitness)),)
        
    except Exception as e:
        print(f"[Adaptive Evaluation Error] {e}")
        return (0.01,)

def constrain_adaptive_individual(individual):
    """
    Ensure adaptive parameters stay within reasonable bounds.
    """
    # risk_reward, trend_strength, entry_threshold, confidence
    individual[0] = max(0.5, min(3.0, individual[0]))   # risk_reward
    individual[1] = max(0.5, min(2.5, individual[1]))   # trend_strength  
    individual[2] = max(0.1, min(2.0, individual[2]))   # entry_threshold
    individual[3] = max(0.5, min(2.5, individual[3]))   # confidence
    
    # buy_threshold_pct, sell_threshold_pct
    individual[4] = max(0.15, min(0.45, individual[4]))  # buy_threshold_pct
    individual[5] = max(0.55, min(0.85, individual[5]))  # sell_threshold_pct
    
    # Ensure buy < sell with minimum gap
    if individual[4] >= individual[5] - 0.15:
        individual[4] = individual[5] - 0.20
        individual[4] = max(0.15, individual[4])
    
    # Regime multipliers (more conservative)
    individual[6] = max(0.9, min(1.3, individual[6]))   # bull_multiplier
    individual[7] = max(0.7, min(1.1, individual[7]))   # bear_multiplier
    individual[8] = max(0.6, min(0.9, individual[8]))   # high_vol_multiplier
    individual[9] = max(1.0, min(1.3, individual[9]))   # low_vol_multiplier
    
    # Risk management (FIXED - critical bounds)
    individual[10] = max(0.5, min(0.85, individual[10])) # max_position_pct (max 85%)
    individual[11] = max(0.05, min(0.12, individual[11])) # stop_loss_pct (5-12%)
    individual[12] = max(0.10, min(0.20, individual[12])) # take_profit_pct (10-20%)
    
    # Ensure take_profit > stop_loss
    if individual[12] <= individual[11]:
        individual[12] = individual[11] + 0.05
        individual[12] = min(0.20, individual[12])
    
    return individual,

def run_adaptive_optimization(df, ngen=25, pop_size=50, seed=None, log=None):
    """
    Optimize adaptive strategy parameters.
    """
    if seed is None:
        seed = int(time.time() * 1000000) % 1000000
    
    random.seed(seed)
    np.random.seed(seed % 2147483647)
    
    if log:
        log.info(f"🧬 [Adaptive] Starting optimization with seed: {seed}")

    # Configure fitness for adaptive strategy
    fitness_config = FitnessConfig(
        min_trades=8,                    # Require meaningful trade frequency
        max_drawdown_threshold=0.5,      # Stricter drawdown control
        min_profit_threshold=-0.10,      # Allow some losses
        profitability_weight=0.35,       # Focus on profits
        risk_adjusted_weight=0.30,       # Risk-adjusted returns
        drawdown_weight=0.25,            # Drawdown control
        trade_quality_weight=0.10        # Trade quality
    )

    toolbox = base.Toolbox()
    
    # Define parameter creation functions
    def create_individual():
        return creator.Individual([
            random.uniform(0.8, 2.5),    # risk_reward (more conservative)
            random.uniform(0.8, 2.0),    # trend_strength
            random.uniform(0.3, 1.5),    # entry_threshold
            random.uniform(0.8, 2.0),    # confidence
            random.uniform(0.20, 0.40),  # buy_threshold_pct (bottom 20-40%)
            random.uniform(0.60, 0.80),  # sell_threshold_pct (top 60-80%)
            random.uniform(1.0, 1.2),    # bull_multiplier (conservative)
            random.uniform(0.8, 1.0),    # bear_multiplier
            random.uniform(0.7, 0.85),   # high_vol_multiplier
            random.uniform(1.1, 1.25),   # low_vol_multiplier
            random.uniform(0.6, 0.8),    # max_position_pct (60-80%)
            random.uniform(0.06, 0.10),  # stop_loss_pct (6-10%)
            random.uniform(0.12, 0.18),  # take_profit_pct (12-18%)
        ])
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("constrain", constrain_adaptive_individual)

    def evaluate(ind):
        return evaluate_adaptive_individual(ind, df, fitness_config)

    toolbox.register("evaluate", evaluate)

    # Create population
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    if log:
        log.info(f"🧬 [Adaptive] Evolution: {ngen} generations, {pop_size} individuals")

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
    
    # Convert to AdaptiveParams
    best_params = AdaptiveParams(
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
    
    # Final evaluation
    best_results = evaluate_adaptive_strategy(df, best_params, fitness_config)
    
    if log:
        final_stats = logbook[-1]
        log.info(f"🧬 [Adaptive] Evolution complete - Max: {final_stats['max']:.4f}, Avg: {final_stats['avg']:.4f}")
        log.info(f"🏆 [Adaptive] Best fitness: {best_results['fitness']:.4f}, Trades: {best_results['trades']}")

    return {
        'params': best_params,
        'fitness': float(best_results['fitness']),
        'results': best_results
    }

# Test the adaptive optimization
def test_adaptive_optimization():
    """
    Test the adaptive optimization system.
    """
    from src.core.data import fetch_historical_data
    from src.core.logging import get_logger
    
    print("🧪 Testing Adaptive Optimization System...")
    
    log = get_logger()
    df = fetch_historical_data(refresh=False)
    
    # Test on recent data
    test_data = df.tail(200)  # Last 200 days
    
    print(f"📊 Testing on {len(test_data)} days of data")
    print(f"   Date range: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Run optimization
    result = run_adaptive_optimization(
        test_data, 
        ngen=8,      # Quick test
        pop_size=20, 
        log=log
    )
    
    print(f"\n✅ Optimization Complete!")
    print(f"   Fitness: {result['fitness']:.4f}")
    print(f"   Trades: {result['results']['trades']}")
    print(f"   Return: {result['results']['returns']:.1%}")
    print(f"   Win Rate: {result['results']['wins'] / max(result['results']['trades'], 1):.1%}")
    
    print(f"\n🎯 Optimized Parameters:")
    params = result['params']
    print(f"   Buy Threshold: {params.buy_threshold_pct:.1%}")
    print(f"   Sell Threshold: {params.sell_threshold_pct:.1%}")
    print(f"   Bull Multiplier: {params.bull_multiplier:.2f}")
    print(f"   Bear Multiplier: {params.bear_multiplier:.2f}")
    print(f"   Stop Loss: {params.stop_loss_pct:.1%}")
    print(f"   Take Profit: {params.take_profit_pct:.1%}")

if __name__ == "__main__":
    test_adaptive_optimization()