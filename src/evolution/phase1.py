# evolution/phase1.py
from deap import base, creator, tools, algorithms
import random
from src.evolution.fitness import calculate_fitness
from src.evolution.utils import get_returns, get_performance_metrics
from src.core.params import Phase1Params
import numpy as np

# Only register DEAP classes once
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluate_phase1(individual, historical_data):
    try:
        params = Phase1Params(*individual)
        returns = get_returns(historical_data)
        score = calculate_fitness(returns)
        return (score,)
    except Exception as e:
        print(f"[Phase1 Error] {e}")
        return (0.0,)

def run_phase1_optimization(historical_data, ngen=10, pop_size=30, seed=None, log=None):
    if seed is not None:
        random.seed(seed)
        if log:
            log.info(f"🎯 [Phase1] Random seed set to {seed}")

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.5, 2.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_phase1, historical_data=historical_data)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=ngen,
        stats=stats, halloffame=hof, verbose=False
    )

    best = hof[0]
    best_params = Phase1Params(*best)
    final_returns = get_returns(historical_data)
    metrics = get_performance_metrics(final_returns)

    return {
        'params': best_params,
        'fitness': float(best.fitness.values[0]),
        'metrics': metrics
    }
