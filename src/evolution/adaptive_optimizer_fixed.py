# src/evolution/adaptive_optimizer_fixed.py

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

def validate_and_fix_parameters(params_list):
    """
    Absolutely ensure parameters are valid - fix any issues.
    """
    # Unpack parameters
    (risk_reward, trend_strength, entry_threshold, confidence,
     buy_threshold_pct, sell_threshold_pct, 
     bull_multiplier, bear_multiplier, high_vol_multiplier, low_vol_multiplier,
     max_position_pct, stop_loss_pct, take_profit_pct) = params_list
    
    # Fix core parameters
    risk_reward = max(0.8, min(2.5, abs(risk_reward)))
    trend_strength = max(0.8, min(2.0, abs(trend_strength)))
    entry_threshold = max(0.3, min(1.5, abs(entry_threshold)))
    confidence = max(0.8, min(2.0, abs(confidence)))
    
    # Fix threshold percentages
    buy_threshold_pct = max(0.15, min(0.45, abs(buy_threshold_pct)))
    sell_threshold_pct = max(0.55, min(0.85, abs(sell_threshold_pct)))
    
    # Ensure proper relationship
    if buy_threshold_pct >= sell_threshold_pct - 0.15:
        buy_threshold_pct = 0.25
        sell_threshold_pct = 0.75
    
    # Fix regime multipliers
    bull_multiplier = max(1.0, min(1.3, abs(bull_multiplier)))
    bear_multiplier = max(0.7, min(1.0, abs(bear_multiplier)))
    high_vol_multiplier = max(0.6, min(0.9, abs(high_vol_multiplier)))
    low_vol_multiplier = max(1.0, min(1.3, abs(low_vol_multiplier)))
    
    # CRITICAL: Fix risk management parameters
    max_position_pct = max(0.5, min(0.8, abs(max_position_pct)))
    stop_loss_pct = max(0.05, min(0.12, abs(stop_loss_pct)))      # 5-12%
    take_profit_pct = max(0.10, min(0.20, abs(take_profit_pct)))  # 10-20%
    
    # Ensure take profit > stop loss
    if take_profit_pct <= stop_loss_pct:
        stop_loss_pct = 0.08   # 8%
        take_profit_pct = 0.15 # 15%
    
    return [
        risk_reward, trend_strength, entry_threshold, confidence,
        buy_threshold_pct, sell_threshold_pct,
        bull_multiplier, bear_multiplier, high_vol_multiplier, low_vol_multiplier,
        max_position_pct, stop_loss_pct, take_profit_pct
    ]

def evaluate_adaptive_individual_fixed(individual, df, fitness_config):
    """
    Evaluate individual with mandatory parameter validation.
    """
    try:
        # MANDATORY: Fix parameters before evaluation
        fixed_params = validate_and_fix_parameters(individual)
        
        # Convert to AdaptiveParams
        params = AdaptiveParams(
            risk_reward=fixed_params[0],
            trend_strength=fixed_params[1],
            entry_threshold=fixed_params[2],
            confidence=fixed_params[3],
            buy_threshold_pct=fixed_params[4],
            sell_threshold_pct=fixed_params[5],
            bull_multiplier=fixed_params[6],
            bear_multiplier=fixed_params[7],
            high_vol_multiplier=fixed_params[8],
            low_vol_multiplier=fixed_params[9],
            max_position_pct=fixed_params[10],
            stop_loss_pct=fixed_params[11],
            take_profit_pct=fixed_params[12]
        )
        
        # Double-check critical parameters
        assert 0.05 <= params.stop_loss_pct <= 0.12, f"Invalid stop loss: {params.stop_loss_pct}"
        assert 0.10 <= params.take_profit_pct <= 0.20, f"Invalid take profit: {params.take_profit_pct}"
        assert params.take_profit_pct > params.stop_loss_pct, "Take profit must be > stop loss"
        assert 0.5 <= params.max_position_pct <= 0.8, f"Invalid position size: {params.max_position_pct}"
        
        # Evaluate strategy
        results = evaluate_adaptive_strategy(df, params, fitness_config)
        fitness = results.get('fitness', 0.01)
        
        # Validation penalties
        trades = results.get('trades', 0)
        max_dd = results.get('max_drawdown', 1.0)
        final_balance = results.get('final_balance', 5000)
        
        if trades == 0:
            return (0.01,)
        
        if max_dd > 0.3:  # More than 30% drawdown
            fitness *= 0.5
        
        if final_balance < 8000:  # Lost more than 20%
            fitness *= 0.7
        
        return (max(0.01, min(1.0, fitness)),)
        
    except Exception as e:
        print(f"[Evaluation Error] {e}")
        return (0.01,)

def create_safe_individual():
    """
    Create individual with guaranteed safe parameters.
    """
    return creator.Individual([
        random.uniform(1.0, 2.0),      # risk_reward
        random.uniform(1.0, 1.8),      # trend_strength
        random.uniform(0.5, 1.2),      # entry_threshold
        random.uniform(1.0, 1.8),      # confidence
        random.uniform(0.20, 0.40),    # buy_threshold_pct
        random.uniform(0.60, 0.80),    # sell_threshold_pct
        random.uniform(1.05, 1.25),    # bull_multiplier
        random.uniform(0.75, 0.95),    # bear_multiplier
        random.uniform(0.65, 0.85),    # high_vol_multiplier
        random.uniform(1.05, 1.25),    # low_vol_multiplier
        random.uniform(0.55, 0.75),    # max_position_pct
        random.uniform(0.06, 0.10),    # stop_loss_pct
        random.uniform(0.12, 0.18),    # take_profit_pct
    ])

def constrain_individual_mandatory(individual):
    """
    Apply mandatory constraints - no exceptions.
    """
    # Fix using our validation function
    fixed_params = validate_and_fix_parameters(individual)
    
    # Update individual in place
    for i in range(len(individual)):
        individual[i] = fixed_params[i]
    
    return individual,

def run_adaptive_optimization_fixed(df, ngen=25, pop_size=50, seed=None, log=None):
    """
    Fixed adaptive optimization with mandatory parameter validation.
    """
    if seed is None:
        seed = int(time.time() * 1000000) % 1000000
    
    random.seed(seed)
    np.random.seed(seed % 2147483647)
    
    if log:
        log.info(f"🔧 [Fixed Adaptive] Starting optimization with seed: {seed}")

    # More restrictive fitness config
    fitness_config = FitnessConfig(
        min_trades=5,
        max_drawdown_threshold=0.3,      # 30% max
        min_profit_threshold=-0.05,      # Allow 5% loss max
        profitability_weight=0.40,
        risk_adjusted_weight=0.30,
        drawdown_weight=0.20,
        trade_quality_weight=0.10
    )

    toolbox = base.Toolbox()
    
    # Use safe individual creation
    toolbox.register("individual", create_safe_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Genetic operators with mandatory constraints
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("constrain", constrain_individual_mandatory)

    def evaluate(ind):
        # Apply constraints before evaluation
        toolbox.constrain(ind)
        return evaluate_adaptive_individual_fixed(ind, df, fitness_config)

    toolbox.register("evaluate", evaluate)

    # Create population with constraints applied
    pop = toolbox.population(n=pop_size)
    for ind in pop:
        toolbox.constrain(ind)
    
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    if log:
        log.info(f"🔧 [Fixed Adaptive] Evolution: {ngen} generations, {pop_size} individuals")

    # Run evolution with constraints applied at each step
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, 
        cxpb=0.6,
        mutpb=0.3,
        ngen=ngen,
        stats=stats, 
        halloffame=hof, 
        verbose=False
    )
    
    # Final constraint application
    for ind in pop:
        toolbox.constrain(ind)

    # Get best result and validate
    best = hof[0]
    toolbox.constrain(best)  # Ensure best is valid
    
    # Create final parameters
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
    
    # Validate final parameters
    print(f"🔍 Final Parameter Validation:")
    print(f"   Stop Loss: {best_params.stop_loss_pct:.1%} (should be 5-12%)")
    print(f"   Take Profit: {best_params.take_profit_pct:.1%} (should be 10-20%)")
    print(f"   Max Position: {best_params.max_position_pct:.1%} (should be 50-80%)")
    print(f"   Take Profit > Stop Loss: {best_params.take_profit_pct > best_params.stop_loss_pct}")
    
    # Final evaluation
    best_results = evaluate_adaptive_strategy(df, best_params, fitness_config)
    
    if log:
        final_stats = logbook[-1]
        log.info(f"🔧 [Fixed Adaptive] Complete - Max: {final_stats['max']:.4f}, Avg: {final_stats['avg']:.4f}")
        log.info(f"🏆 [Fixed Adaptive] Best fitness: {best_results['fitness']:.4f}, Trades: {best_results['trades']}")

    return {
        'params': best_params,
        'fitness': float(best_results['fitness']),
        'results': best_results
    }

# Test function
def test_fixed_optimization():
    """
    Test the fixed optimization system.
    """
    from src.core.data import fetch_historical_data
    from src.core.logging import get_logger
    
    print("🔧 Testing FIXED Adaptive Optimization...")
    
    log = get_logger()
    df = fetch_historical_data(refresh=False)
    
    # Test on training data
    train_data = df[df.index <= '2023-12-31']
    
    print(f"📊 Testing on {len(train_data)} days of training data")
    
    # Run fixed optimization
    result = run_adaptive_optimization_fixed(
        train_data, 
        ngen=10,     # Quick test
        pop_size=25, 
        log=log
    )
    
    print(f"\n✅ Fixed Optimization Complete!")
    print(f"   Fitness: {result['fitness']:.4f}")
    print(f"   Trades: {result['results']['trades']}")
    print(f"   Return: {result['results']['returns']:.1%}")
    print(f"   Stop Loss: {result['params'].stop_loss_pct:.1%}")
    print(f"   Take Profit: {result['params'].take_profit_pct:.1%}")

if __name__ == "__main__":
    test_fixed_optimization()