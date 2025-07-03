# src/evolution/fitness.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Tuple, List
import warnings
from src.core.performance import get_performance_metrics
from src.core.params import Phase2Params
from src.evolution.utils import get_returns

# DEAP imports for advanced evolutionary algorithms
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    warnings.warn("DEAP not available. Install with: pip install deap")


@dataclass
class FitnessConfig:
    """Configuration for fitness calculation parameters."""
    # Component weights
    profitability_weight: float = 0.40  # Actual profit/loss
    risk_adjusted_weight: float = 0.25  # Sharpe/Calmar ratios
    drawdown_weight: float = 0.20       # Maximum drawdown penalty
    trade_quality_weight: float = 0.15  # Trade frequency and win rate
    
    # Scaling factors
    sharpe_scale: float = 3.0
    calmar_scale: float = 2.0
    volatility_scale: float = 2.5
    drawdown_scale: float = 4.0
    
    # Trade validation
    min_trades: int = 5
    min_win_rate: float = 0.30
    max_trades_per_period: int = 100
    
    # Risk management
    max_drawdown_threshold: float = 0.50
    min_profit_threshold: float = -0.20  # Allow some loss but not too much
    
    # Data quality
    min_periods: int = 30
    outlier_clip_pct: float = 0.01
    
    # Minimum guarantees
    min_fitness: float = 0.01
    no_trade_fitness: float = 0.005


@dataclass
class BacktestResult:
    """Container for backtest results."""
    final_balance: float
    initial_balance: float = 10000.0
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    returns: pd.Series = None
    trades_executed: List[Dict] = None
    
    @property
    def total_return(self) -> float:
        return (self.final_balance - self.initial_balance) / self.initial_balance
    
    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.winning_trades / self.trade_count
    
    @property
    def profit_factor(self) -> float:
        if self.losing_trades == 0:
            return float('inf') if self.winning_trades > 0 else 1.0
        # Simplified profit factor approximation
        return max(0.1, self.win_rate / (1 - self.win_rate + 0.01))


def calculate_backtest_fitness(
    backtest_result: BacktestResult,
    config: Optional[FitnessConfig] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate fitness based on actual backtest results.
    
    This is the primary fitness function that should be used with real trading results.
    
    Args:
        backtest_result: Results from running the strategy backtest
        config: Configuration parameters
        
    Returns:
        Tuple of (fitness_score, component_breakdown)
    """
    config = config or FitnessConfig()
    
    # Initialize components
    components = {
        'profitability': 0.0,
        'risk_adjusted': 0.0,
        'drawdown': 0.0,
        'trade_quality': 0.0,
        'total_trades': backtest_result.trade_count,
        'win_rate': backtest_result.win_rate,
        'total_return': backtest_result.total_return
    }
    
    # 1. Handle no-trade scenarios
    if backtest_result.trade_count == 0:
        return config.no_trade_fitness, components
    
    # 2. Hard constraints - immediate disqualification
    if (backtest_result.max_drawdown > config.max_drawdown_threshold or
        backtest_result.total_return < config.min_profit_threshold):
        return config.min_fitness, components
    
    # 3. Profitability component (normalized total return)
    total_return = backtest_result.total_return
    profitability_raw = np.tanh(total_return * 2)  # Scale returns
    profitability_component = max(0.0, profitability_raw)
    components['profitability'] = profitability_component
    
    # 4. Risk-adjusted component
    if backtest_result.returns is not None and len(backtest_result.returns) > 0:
        try:
            returns_clean = backtest_result.returns.replace([np.inf, -np.inf], 0).fillna(0)
            volatility = returns_clean.std()
            
            if volatility > 0:
                sharpe = np.sqrt(252) * returns_clean.mean() / volatility
                calmar = total_return * 252 / max(backtest_result.max_drawdown, 0.01)
                
                sharpe_norm = np.tanh(sharpe / config.sharpe_scale)
                calmar_norm = np.tanh(calmar / config.calmar_scale)
                risk_adjusted_component = (sharpe_norm + calmar_norm) / 2
            else:
                risk_adjusted_component = 0.0
        except Exception:
            risk_adjusted_component = 0.0
    else:
        # Fallback risk adjustment based on return/drawdown ratio
        risk_adjusted_component = np.tanh(total_return / max(backtest_result.max_drawdown, 0.01))
    
    components['risk_adjusted'] = max(0.0, risk_adjusted_component)
    
    # 5. Drawdown component (penalty)
    drawdown_component = 1.0 - np.tanh(backtest_result.max_drawdown * config.drawdown_scale)
    components['drawdown'] = max(0.0, drawdown_component)
    
    # 6. Trade quality component
    trade_frequency_score = min(1.0, backtest_result.trade_count / config.min_trades)
    win_rate_score = max(0.0, (backtest_result.win_rate - config.min_win_rate) / (1.0 - config.min_win_rate))
    profit_factor_score = np.tanh(backtest_result.profit_factor / 2.0)
    
    # Penalize over-trading
    if backtest_result.trade_count > config.max_trades_per_period:
        overtrading_penalty = config.max_trades_per_period / backtest_result.trade_count
    else:
        overtrading_penalty = 1.0
    
    trade_quality_component = (trade_frequency_score * win_rate_score * profit_factor_score * overtrading_penalty)
    components['trade_quality'] = trade_quality_component
    
    # 7. Weighted combination
    fitness_raw = (
        config.profitability_weight * profitability_component +
        config.risk_adjusted_weight * risk_adjusted_component +
        config.drawdown_weight * drawdown_component +
        config.trade_quality_weight * trade_quality_component
    )
    
    # 8. Final bounds and rounding
    fitness = max(config.min_fitness, min(1.0, fitness_raw))
    
    return round(fitness, 6), components


def calculate_fitness(
    returns: pd.Series, 
    risk_free_rate: float = 0.02 / 252,
    config: Optional[FitnessConfig] = None
) -> float:
    """
    Legacy fitness function for returns-only scenarios.
    
    NOTE: This should be phased out in favor of calculate_backtest_fitness()
    for actual strategy evaluation.
    """
    config = config or FitnessConfig()
    
    # Input validation
    if returns is None or len(returns) == 0 or returns.isna().all():
        return config.no_trade_fitness
    
    if len(returns) < config.min_periods:
        return config.min_fitness
    
    # Create a mock backtest result for compatibility
    mock_result = BacktestResult(
        final_balance=10000 * (1 + returns.sum()),
        trade_count=max(5, len(returns) // 10),  # Estimate trades
        winning_trades=max(1, int(len(returns[returns > 0]) * 0.6)),
        losing_trades=max(1, len(returns) - int(len(returns[returns > 0]) * 0.6)),
        max_drawdown=calculate_max_drawdown(returns),
        returns=returns
    )
    
    fitness, _ = calculate_backtest_fitness(mock_result, config)
    return fitness


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Helper function to calculate maximum drawdown from returns."""
    try:
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (peak - cumulative) / peak
        return float(drawdown.max())
    except Exception:
        return 0.0


def evaluate_strategy_with_backtest(
    df: pd.DataFrame,
    params: Phase2Params,
    backtest_func: Callable,
    config: Optional[FitnessConfig] = None
) -> Dict[str, Any]:
    """
    Evaluate strategy using actual backtest results.
    
    Args:
        df: Historical data
        params: Strategy parameters
        backtest_func: Function that runs backtest and returns BacktestResult
        config: Fitness configuration
        
    Returns:
        Dictionary with fitness and detailed metrics
    """
    try:
        # Run the actual backtest
        backtest_result = backtest_func(df, params)
        
        # Calculate fitness from backtest results
        fitness, components = calculate_backtest_fitness(backtest_result, config)
        
        # Combine with traditional metrics if available
        metrics = {
            'fitness': fitness,
            'total_return': backtest_result.total_return,
            'final_balance': backtest_result.final_balance,
            'trade_count': backtest_result.trade_count,
            'win_rate': backtest_result.win_rate,
            'max_drawdown': backtest_result.max_drawdown,
            'profit_factor': backtest_result.profit_factor,
            **components  # Include component breakdown
        }
        
        # Add traditional performance metrics if returns available
        if backtest_result.returns is not None:
            try:
                traditional_metrics = get_performance_metrics(backtest_result.returns)
                metrics.update(traditional_metrics)
            except Exception as e:
                print(f"[Traditional Metrics Warning] {e}")
        
        return metrics
        
    except Exception as e:
        print(f"[Strategy Evaluation Error] {e}")
        return {
            'fitness': config.min_fitness if config else 0.01,
            'total_return': -1.0,
            'final_balance': 0.0,
            'trade_count': 0,
            'win_rate': 0.0,
            'max_drawdown': 1.0,
            'profit_factor': 0.0,
            'error': str(e)
        }


# ============================================================================
# DEAP Integration for Advanced Evolution
# ============================================================================

if DEAP_AVAILABLE:
    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    def create_deap_toolbox(
        param_bounds: Dict[str, Tuple[float, float]],
        fitness_func: Callable,
        config: Optional[FitnessConfig] = None
    ) -> base.Toolbox:
        """
        Create DEAP toolbox for parameter optimization.
        
        Args:
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
            fitness_func: Function that takes parameters and returns fitness
            config: Fitness configuration
            
        Returns:
            Configured DEAP toolbox
        """
        toolbox = base.Toolbox()
        
        # Parameter generation functions
        param_names = list(param_bounds.keys())
        
        def create_individual():
            individual = []
            for param_name in param_names:
                min_val, max_val = param_bounds[param_name]
                value = np.random.uniform(min_val, max_val)
                individual.append(value)
            return creator.Individual(individual)
        
        def evaluate_individual(individual):
            # Convert individual to parameter dictionary
            params = {name: value for name, value in zip(param_names, individual)}
            try:
                fitness = fitness_func(params)
                return (fitness,)
            except Exception as e:
                print(f"[DEAP Evaluation Error] {e}")
                return (config.min_fitness if config else 0.01,)
        
        def mutate_individual(individual, mu=0, sigma=0.1, indpb=0.2):
            """Gaussian mutation with parameter bounds checking."""
            for i in range(len(individual)):
                if np.random.random() < indpb:
                    param_name = param_names[i]
                    min_val, max_val = param_bounds[param_name]
                    
                    # Gaussian mutation
                    mutation = np.random.normal(mu, sigma)
                    individual[i] += mutation * (max_val - min_val)
                    
                    # Ensure bounds
                    individual[i] = np.clip(individual[i], min_val, max_val)
            
            return individual,
        
        def crossover_individuals(ind1, ind2, alpha=0.5):
            """Blend crossover with bounds checking."""
            for i in range(len(ind1)):
                param_name = param_names[i]
                min_val, max_val = param_bounds[param_name]
                
                # Blend crossover
                gamma = (1 + 2 * alpha) * np.random.random() - alpha
                
                new_val1 = (1 - gamma) * ind1[i] + gamma * ind2[i]
                new_val2 = gamma * ind1[i] + (1 - gamma) * ind2[i]
                
                # Ensure bounds
                ind1[i] = np.clip(new_val1, min_val, max_val)
                ind2[i] = np.clip(new_val2, min_val, max_val)
            
            return ind1, ind2
        
        # Register functions
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", crossover_individuals)
        toolbox.register("mutate", mutate_individual)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        return toolbox

    def optimize_with_deap(
        param_bounds: Dict[str, Tuple[float, float]],
        fitness_func: Callable,
        population_size: int = 50,
        generations: int = 20,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        config: Optional[FitnessConfig] = None,
        verbose: bool = True
    ) -> Tuple[Dict[str, float], float, List[Dict]]:
        """
        Optimize parameters using DEAP genetic algorithm.
        
        Args:
            param_bounds: Parameter bounds dictionary
            fitness_func: Fitness evaluation function
            population_size: Size of the population
            generations: Number of generations
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            config: Fitness configuration
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_params, best_fitness, generation_stats)
        """
        if not DEAP_AVAILABLE:
            raise ImportError("DEAP is required for advanced optimization. Install with: pip install deap")
        
        toolbox = create_deap_toolbox(param_bounds, fitness_func, config)
        
        # Statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Hall of fame to track best individuals
        hof = tools.HallOfFame(1)
        
        # Create initial population
        population = toolbox.population(n=population_size)
        
        # Run evolution
        if verbose:
            print(f"🧬 Starting DEAP optimization: {generations} generations, {population_size} individuals")
        
        population, logbook = algorithms.eaSimple(
            population, toolbox,
            cxpb=crossover_prob,
            mutpb=mutation_prob,
            ngen=generations,
            stats=stats,
            halloffame=hof,
            verbose=verbose
        )
        
        # Extract best individual
        best_individual = hof[0]
        param_names = list(param_bounds.keys())
        best_params = {name: value for name, value in zip(param_names, best_individual)}
        best_fitness = best_individual.fitness.values[0]
        
        # Convert logbook to list of dictionaries
        generation_stats = []
        for record in logbook:
            generation_stats.append(dict(record))
        
        if verbose:
            print(f"🏆 Best fitness: {best_fitness:.6f}")
            print(f"🎯 Best parameters: {best_params}")
        
        return best_params, best_fitness, generation_stats

else:
    def optimize_with_deap(*args, **kwargs):
        raise ImportError("DEAP is not available. Install with: pip install deap")


# ============================================================================
# Testing and Example Usage
# ============================================================================

def mock_backtest_function(df: pd.DataFrame, params: Phase2Params) -> BacktestResult:
    """Mock backtest function for testing purposes."""
    # Simulate some trading results
    np.random.seed(42)
    trade_count = np.random.randint(5, 25)
    win_rate = np.random.uniform(0.4, 0.7)
    total_return = np.random.normal(0.1, 0.3)
    max_dd = np.random.uniform(0.05, 0.25)
    
    winning_trades = int(trade_count * win_rate)
    losing_trades = trade_count - winning_trades
    
    # Generate some mock returns
    returns = pd.Series(np.random.normal(total_return/252, 0.02, 252))
    
    return BacktestResult(
        final_balance=10000 * (1 + total_return),
        trade_count=trade_count,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        max_drawdown=max_dd,
        returns=returns
    )


if __name__ == "__main__":
    # Test the new fitness function
    print("🧪 Testing Enhanced Fitness Function")
    
    # Create mock backtest result
    mock_result = BacktestResult(
        final_balance=12500.0,  # 25% return
        trade_count=15,
        winning_trades=10,
        losing_trades=5,
        max_drawdown=0.15,
        returns=pd.Series(np.random.normal(0.001, 0.02, 252))
    )
    
    fitness, components = calculate_backtest_fitness(mock_result)
    print(f"\n📊 Mock Strategy Results:")
    print(f"   Fitness: {fitness:.4f}")
    print(f"   Components: {components}")
    
    # Test DEAP optimization if available
    if DEAP_AVAILABLE:
        print(f"\n🧬 Testing DEAP Optimization:")
        
        # Define parameter bounds for a mock optimization
        param_bounds = {
            'risk_reward': (0.5, 3.0),
            'trend': (0.5, 2.5),
            'entry': (0.1, 2.0),
            'confidence': (0.5, 2.5)
        }
        
        def mock_fitness_func(params):
            # Mock fitness function that prefers balanced parameters
            target = {'risk_reward': 1.5, 'trend': 1.2, 'entry': 0.8, 'confidence': 1.0}
            
            # Calculate distance from target
            distance = sum((params[k] - target[k])**2 for k in params.keys())
            fitness = np.exp(-distance)  # Higher fitness for closer to target
            
            return fitness
        
        try:
            best_params, best_fitness, stats = optimize_with_deap(
                param_bounds=param_bounds,
                fitness_func=mock_fitness_func,
                population_size=20,
                generations=5,
                verbose=True
            )
            
            print(f"\n🏆 DEAP Results:")
            print(f"   Best Fitness: {best_fitness:.6f}")
            print(f"   Best Params: {best_params}")
            
        except Exception as e:
            print(f"   DEAP test failed: {e}")
    
    else:
        print(f"\n⚠️  DEAP not available - install with: pip install deap")