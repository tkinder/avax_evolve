# src/evolution/phase2_fixed.py
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from deap import base, creator, tools, algorithms
import random
import numpy as np
import time
from src.core.backtester import evaluate_strategy_with_new_fitness
from src.evolution.fitness import FitnessConfig
from src.core.params import Phase2Params

def improved_seed_generator():
    """Generate truly random seeds with good separation"""
    base_seed = int(time.time() * 1000000) % 1000000
    offset = random.randint(10000, 99999)
    return base_seed + offset

def evaluate_phase2_individual(individual, df, base_params, fitness_config):
    """
    Evaluate Phase2 individual using ACTUAL backtesting.
    """
    try:
        bullish, bearish, top, bottom, neutral = individual
        
        # Parameter validation - reject clearly invalid combinations
        if top < 0 or bottom < 0:  # Negative price levels are invalid
            return (0.01,)
        
        if abs(top - bottom) < 0.1:  # Too narrow trading range
            return (0.01,)
            
        if any(abs(x) > 10 for x in individual):  # Extreme parameter values
            return (0.01,)
        
        # Create combined parameters
        combined_params = Phase2Params(
            bullish=bullish,
            bearish=bearish,
            top=top,
            bottom=bottom,
            neutral=neutral,
            risk_reward=base_params.risk_reward,
            trend=base_params.trend,
            entry=base_params.entry,
            confidence=base_params.confidence
        )
        
        # Use the new backtest-driven evaluation
        metrics = evaluate_strategy_with_new_fitness(df, combined_params, fitness_config)
        fitness = metrics.get('fitness', 0.01)
        
        # Additional Phase2 validations
        trade_count = metrics.get('trade_count', 0)
        max_dd = metrics.get('max_drawdown', 1.0)
        final_balance = metrics.get('final_balance', 10000)
        
        # Heavy penalties for bad outcomes
        if trade_count == 0:
            return (0.005,)  # Even lower for no trades in Phase2
        
        if max_dd > 0.8:  # Catastrophic drawdown
            return (0.005,)
            
        if final_balance < 5000:  # Lost more than half
            return (max(0.01, fitness * 0.1),)
        
        return (max(0.01, min(1.0, fitness)),)
        
    except Exception as e:
        print(f"[Phase2 Evaluation Error] {e}")
        return (0.01,)

def constrain_individual(individual):
    """Ensure individual parameters stay within reasonable bounds"""
    # bullish, bearish, top, bottom, neutral
    individual[0] = max(-1.0, min(3.0, individual[0]))  # bullish
    individual[1] = max(-1.0, min(1.0, individual[1]))  # bearish  
    individual[2] = max(0.1, min(5.0, individual[2]))   # top
    individual[3] = max(0.1, min(5.0, individual[3]))   # bottom
    individual[4] = max(-1.0, min(3.0, individual[4]))  # neutral
    
    # Ensure top > bottom (swap if needed)
    if individual[2] < individual[3]:
        individual[2], individual[3] = individual[3], individual[2]
    
    return individual,

def run_phase2_optimization(df, base_params, ngen=25, pop_size=60, seed=None, log=None):
    """
    Run Phase2 optimization using actual backtest fitness.
    """
    # Use improved seeding
    if seed is None:
        seed = improved_seed_generator()
    
    random.seed(seed)
    np.random.seed(seed % 2147483647)
    
    if log:
        log.info(f"🎯 [Phase2] Using improved seed: {seed}")

    # Ensure DEAP classes exist
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # Configure fitness for Phase2 (more demanding)
    fitness_config = FitnessConfig(
        min_trades=5,  # Require reasonable trade frequency
        max_drawdown_threshold=0.4,  # Stricter drawdown control
        min_profit_threshold=-0.15,  # Limit acceptable losses
        profitability_weight=0.4,   # Focus on actual profits
        risk_adjusted_weight=0.25,  
        drawdown_weight=0.25,       # Penalize high drawdowns
        trade_quality_weight=0.1
    )

    toolbox = base.Toolbox()
    
    # More constrained parameter ranges based on what makes sense
    def create_individual():
        return creator.Individual([
            random.uniform(-0.5, 2.0),  # bullish: can be slightly negative to positive
            random.uniform(-1.0, 0.5),  # bearish: should be negative to neutral
            random.uniform(0.5, 4.0),   # top: positive price multiplier
            random.uniform(0.1, 3.0),   # bottom: positive price multiplier  
            random.uniform(-0.5, 2.0)   # neutral: slightly negative to positive
        ])
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Improved genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("constrain", constrain_individual)

    def evaluate(ind):
        return evaluate_phase2_individual(ind, df, base_params, fitness_config)

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
        log.info(f"🧬 [Phase2] Starting evolution: {ngen} generations, {pop_size} individuals")

    # Run evolution with constraint handling
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
    bullish, bearish, top, bottom, neutral = best
    
    best_params = Phase2Params(
        bullish=bullish,
        bearish=bearish,
        top=top,
        bottom=bottom,
        neutral=neutral,
        risk_reward=base_params.risk_reward,
        trend=base_params.trend,
        entry=base_params.entry,
        confidence=base_params.confidence
    )
    
    # Final evaluation with detailed metrics
    best_metrics = evaluate_strategy_with_new_fitness(df, best_params, fitness_config)
    
    if log:
        final_stats = logbook[-1]
        log.info(f"🧬 [Phase2] Evolution Stats - Max: {final_stats['max']:.4f}, Avg: {final_stats['avg']:.4f}")
        log.info(f"🏆 [Phase2] Best fitness: {best_metrics.get('fitness', 0.0):.4f}, Trades: {best_metrics.get('trade_count', 0)}")

    return {
        'params': best_params,
        'fitness': float(best_metrics.get('fitness', 0.0)),
        'metrics': best_metrics
    }