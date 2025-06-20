# src/evolution/phase1_fixed.py
from deap import base, creator, tools, algorithms
import random
import numpy as np
from src.core.backtester import evaluate_strategy_with_new_fitness, FitnessConfig
from core.params import Phase1Params

# Ensure DEAP classes are created only once
def _init_deap():
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

_init_deap()

def evaluate_phase1(individual, historical_data, fitness_config):
    """
    Evaluate Phase1 params using ACTUAL backtesting instead of synthetic returns.
    """
    try:
        # Convert individual to Phase1Params
        risk_reward, trend, entry, confidence = individual
        
        # Validate parameters - reject obviously bad values
        if any(x <= 0 for x in individual):
            return (0.01,)  # Minimum fitness for invalid params
        
        # Create minimal Phase2Params for backtesting (Phase1 is subset)
        from core.params import Phase2Params
        params = Phase2Params(
            risk_reward=risk_reward,
            trend=trend, 
            entry=entry,
            confidence=confidence,
            # Use neutral defaults for Phase2-specific params
            bullish=1.0,
            bearish=1.0,
            top=1.5,     # Will be converted to reasonable price levels
            bottom=0.5,  # Will be converted to reasonable price levels  
            neutral=1.0
        )
        
        # Use the new backtest-driven fitness evaluation
        metrics = evaluate_strategy_with_new_fitness(historical_data, params, fitness_config)
        fitness = metrics.get('fitness', 0.01)
        
        # Extra validation - reject strategies with no trades or extreme results
        trade_count = metrics.get('trade_count', 0)
        max_dd = metrics.get('max_drawdown', 1.0)
        
        if trade_count == 0:
            fitness *= 0.1  # Heavy penalty for no trades
        if max_dd > 0.5:  # More than 50% drawdown
            fitness *= 0.1  # Heavy penalty for excessive risk
            
        return (max(0.01, min(1.0, fitness)),)
        
    except Exception as e:
        print(f"[Phase1 Evaluation Error] {e}")
        return (0.01,)

def run_phase1_optimization(historical_data, ngen=20, pop_size=50, seed=None, log=None):
    """
    Run Phase1 optimization using actual backtest fitness.
    """
    if seed is not None:
        random.seed(seed)
        if log:
            log.info(f"🎯 [Phase1] Random seed set to {seed}")

    # Configure fitness for Phase1 (more conservative)
    fitness_config = FitnessConfig(
        min_trades=3,  # Lower requirement for Phase1
        max_drawdown_threshold=0.6,  # More lenient for Phase1
        min_profit_threshold=-0.3,  # Allow some losses in Phase1
        profitability_weight=0.3,   # Less focus on pure profit
        risk_adjusted_weight=0.4,   # More focus on risk-adjusted returns
        drawdown_weight=0.2,
        trade_quality_weight=0.1
    )

    toolbox = base.Toolbox()
    # Tighter parameter ranges for Phase1
    toolbox.register("attr_rr", random.uniform, 0.5, 3.0)     # risk_reward
    toolbox.register("attr_trend", random.uniform, 0.5, 2.5)  # trend
    toolbox.register("attr_entry", random.uniform, 0.1, 2.0)  # entry
    toolbox.register("attr_conf", random.uniform, 0.5, 2.5)   # confidence
    
    def create_individual():
        return creator.Individual([
            toolbox.attr_rr(),
            toolbox.attr_trend(), 
            toolbox.attr_entry(),
            toolbox.attr_conf()
        ])
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_phase1, historical_data=historical_data, fitness_config=fitness_config)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    if log:
        log.info(f"🧬 [Phase1] Starting evolution: {ngen} generations, {pop_size} individuals")

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.6, mutpb=0.3,
        ngen=ngen,
        stats=stats, halloffame=hof,
        verbose=False
    )

    best = hof[0]
    best_params = Phase1Params(*best)
    best_fitness = float(best.fitness.values[0])
    
    if log:
        final_stats = logbook[-1]
        log.info(f"🏆 [Phase1] Best fitness: {best_fitness:.4f}, Final avg: {final_stats['avg']:.4f}")

    return {
        'params': best_params,
        'fitness': best_fitness
    }