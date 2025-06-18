# evolution/phase2_improved.py
from deap import base, creator, tools, algorithms
import random
import numpy as np
import pandas as pd
import time
from src.evolution.fitness import calculate_fitness
from src.core.params import Phase2Params

def improved_seed_generator():
    """Generate truly random seeds with good separation"""
    base_seed = int(time.time() * 1000000) % 1000000  # Microsecond precision
    offset = random.randint(10000, 99999)  # Large random offset
    return base_seed + offset

def backtest_strategy(df, params):
    """
    Simple momentum/mean reversion strategy using the optimized parameters.
    This replaces the placeholder strategy with actual trading logic.
    """
    try:
        if len(df) < 50:  # Need enough data
            return pd.Series([0.0] * len(df))
        
        # Calculate indicators
        close = df['close'].values
        returns = np.zeros(len(close))
        position = 0  # 0 = no position, 1 = long, -1 = short
        
        # Simple moving averages for trend detection
        short_window = 10
        long_window = 30
        
        if len(close) <= long_window:
            return pd.Series(returns)
        
        short_ma = pd.Series(close).rolling(window=short_window).mean()
        long_ma = pd.Series(close).rolling(window=long_window).mean()
        
        # Use parameters to adjust strategy
        risk_reward = getattr(params, 'risk_reward', 1.0)
        trend_strength = getattr(params, 'trend', 1.0)
        entry_threshold = getattr(params, 'entry', 1.0)
        confidence = getattr(params, 'confidence', 1.0)
        
        # Market regime parameters (Phase 2 specific)
        bullish_mult = getattr(params, 'bullish', 1.0)
        bearish_mult = getattr(params, 'bearish', 1.0)
        neutral_mult = getattr(params, 'neutral', 1.0)
        
        # Simple strategy logic
        for i in range(long_window, len(close)):
            # Determine market regime
            trend_strength_current = (short_ma.iloc[i] - long_ma.iloc[i]) / long_ma.iloc[i]
            
            if trend_strength_current > 0.02:  # Bullish
                regime_mult = bullish_mult
            elif trend_strength_current < -0.02:  # Bearish
                regime_mult = bearish_mult
            else:  # Neutral
                regime_mult = neutral_mult
            
            # Entry signals adjusted by parameters
            if position == 0:  # No position
                if short_ma.iloc[i] > long_ma.iloc[i] * (1 + entry_threshold * 0.01):
                    position = 1  # Go long
                elif short_ma.iloc[i] < long_ma.iloc[i] * (1 - entry_threshold * 0.01):
                    position = -1  # Go short
            
            # Calculate returns when in position
            if position != 0 and i > 0:
                price_return = (close[i] - close[i-1]) / close[i-1]
                strategy_return = position * price_return * regime_mult * risk_reward * confidence
                returns[i] = strategy_return
                
                # Exit conditions (simple stop loss/take profit)
                if abs(strategy_return) > 0.05:  # 5% move
                    position = 0
        
        return pd.Series(returns)
    
    except Exception as e:
        print(f"[Backtest Error] {e}")
        return pd.Series([0.0] * len(df))

def get_performance_metrics(returns):
    """
    Calculate performance metrics from returns series.
    """
    try:
        returns = returns.replace([np.inf, -np.inf], 0).fillna(0)
        
        if len(returns) == 0 or returns.std() == 0:
            return {
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'total_return': 0.0,
                'win_rate': 0.0
            }
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate drawdown
        peak = cumulative.cummax()
        drawdown = (peak - cumulative) / peak
        
        # Calculate metrics
        total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0.0
        volatility = returns.std() * np.sqrt(252)  # Annualized
        sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0.0
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0
        
        # Win rate
        positive_returns = returns[returns > 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0
        
        return {
            'sharpe': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'volatility': float(volatility),
            'total_return': float(total_return),
            'win_rate': float(win_rate)
        }
    
    except Exception as e:
        print(f"[Metrics Error] {e}")
        return {
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'total_return': 0.0,
            'win_rate': 0.0
        }

def evaluate_strategy(df, params):
    """
    Evaluate trading strategy using actual backtesting instead of simple returns.
    """
    try:
        # Use actual strategy backtesting
        returns = backtest_strategy(df, params)
        fitness = calculate_fitness(returns)
        metrics = get_performance_metrics(returns)
        
        metrics['fitness'] = fitness
        
        # Convert returns to list for storage
        if hasattr(returns, 'tolist'):
            metrics['returns'] = returns.tolist()
        else:
            metrics['returns'] = list(returns) if returns is not None else []
        
        return metrics
    
    except Exception as e:
        print(f"[Phase2 Strategy Error] {e}")
        return {
            'fitness': 0.0, 
            'returns': [],
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0
        }

def evaluate_phase2_individual(individual, df, base_params):
    try:
        bullish, bearish, top, bottom, neutral = individual
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
        metrics = evaluate_strategy(df, combined_params)
        return (metrics['fitness'],)
    except Exception as e:
        print(f"[Phase2 Evaluation Error] {e}")
        return (0.0,)

def constrain_individual(individual):
    """Ensure individual parameters stay within bounds after mutation"""
    for i in range(len(individual)):
        individual[i] = max(0.1, min(3.0, individual[i]))  # Wider bounds: 0.1 to 3.0
    return individual,

def run_phase2_optimization(df, base_params, ngen=25, pop_size=60, seed=None, log=None):
    # Use improved seeding
    if seed is None:
        seed = improved_seed_generator()
    
    random.seed(seed)
    np.random.seed(seed % 2147483647)  # Ensure numpy uses same randomization
    
    if log:
        log.info(f"🎯 [Phase2] Using improved seed: {seed}")

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    # Wider parameter ranges for better exploration
    toolbox.register("attr_float", random.uniform, 0.1, 3.0)  # Wider range
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Improved genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.4)  # More mutation
    toolbox.register("select", tools.selTournament, tournsize=4)  # Larger tournament
    toolbox.register("constrain", constrain_individual)

    def evaluate(ind):
        return evaluate_phase2_individual(ind, df, base_params)

    toolbox.register("evaluate", evaluate)

    # Larger, more diverse population
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run evolution with constraint handling
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, 
        cxpb=0.7,  # Higher crossover probability
        mutpb=0.4,  # Higher mutation probability  
        ngen=ngen,
        stats=stats, 
        halloffame=hof, 
        verbose=False
    )
    
    # Apply constraints to final population
    for ind in pop:
        toolbox.constrain(ind)

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
    best_metrics = evaluate_strategy(df, best_params)

    # Format metrics for logging
    formatted_metrics = {}
    for k, v in best_metrics.items():
        if isinstance(v, list):
            formatted_metrics[k] = round(np.mean(v), 4) if v else 0.0
        else:
            formatted_metrics[k] = round(float(v), 4) if isinstance(v, (int, float)) else v

    if log:
        # Log evolution statistics
        final_stats = logbook[-1]
        log.info(f"🧬 [Phase2] Evolution Stats - Max: {final_stats['max']:.4f}, Avg: {final_stats['avg']:.4f}, Std: {final_stats['std']:.4f}")

    return {
        'params': best_params,
        'fitness': float(formatted_metrics.get('fitness', 0.0)),
        'metrics': formatted_metrics
    }