#!/usr/bin/env python3
"""
Bitcoin Rolling Validation - Based on AVAX's Proven Approach
Implements proper train/test split validation with much longer time periods
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
from datetime import datetime, timedelta
import requests

# Import Bitcoin components
from bitcoin_backtester import BitcoinParams, evaluate_bitcoin_strategy_with_backtest
from src.evolution.fitness import FitnessConfig

# Import DEAP for optimization
from deap import base, creator, tools, algorithms
import random

class BitcoinRollingValidator:
    """
    Bitcoin Rolling Validation using AVAX's proven approach but with much longer time periods
    """
    
    def __init__(self):
        self.results_history = []
        print("₿ Bitcoin Rolling Validation System Initialized")
        print("🎯 Using AVAX's proven validation approach with extended timeframes")
        
    def fetch_extended_bitcoin_data(self, days=1080):
        """
        Fetch extended Bitcoin data (3+ years like AVAX)
        Using CryptoCompare API with the same key as AVAX
        """
        print(f"📊 Fetching {days} days of Bitcoin historical data...")
        
        # Use CryptoCompare API (same as AVAX)
        API_KEY = '4c322523f98c6c20dbe789194197dafac7329ec5a7dc378503118d443e867c2b'
        
        end_date = pd.Timestamp.today() - pd.Timedelta(days=1)
        url = (
            f"https://min-api.cryptocompare.com/data/v2/histoday?"
            f"fsym=BTC&tsym=USD&limit={days}&toTs={int(end_date.timestamp())}&api_key={API_KEY}"
        )
        
        try:
            print(f"   Fetching Bitcoin data from CryptoCompare...")
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()['Data']['Data']
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            df = df.rename(columns={
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'open': 'open',
                'volumefrom': 'volumefrom',
                'volumeto': 'volumeto'
            })
            
            numeric_columns = ['close', 'high', 'low', 'open', 'volumefrom', 'volumeto']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            print(f"✅ Bitcoin data ready: {len(df)} days from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
            print(f"   Current price: ${df['close'].iloc[-1]:,.0f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching Bitcoin data from CryptoCompare: {e}")
            raise ValueError(f"Failed to fetch Bitcoin data: {e}")
    
    def create_bitcoin_test_periods(self, df, train_end_year=2023):
        """
        Create test periods similar to AVAX but adapted for Bitcoin
        Train on data up to train_end_year, test on subsequent periods
        """
        # Check if we have data before train_end_year
        earliest_date = df.index[0]
        latest_date = df.index[-1]
        
        print(f"   Data range: {earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
        
        # If we don't have data before 2024, adjust the split
        if earliest_date.year >= 2024:
            print(f"   ⚠️  No data before 2024, using first 60% for training, last 40% for testing")
            total_days = len(df)
            train_days = int(total_days * 0.6)
            
            train_data = df.iloc[:train_days].copy()
            test_data = df.iloc[train_days:].copy()
            
            # Create test periods from available test data
            test_periods = []
            chunk_size = len(test_data) // 4  # Split into 4 periods
            
            if chunk_size > 10:  # Only if we have enough data
                for i in range(4):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size if i < 3 else len(test_data)
                    
                    if start_idx < len(test_data):
                        period_data = test_data.iloc[start_idx:end_idx]
                        if len(period_data) > 10:
                            test_periods.append({
                                'name': f'Test_P{i+1}',
                                'start': period_data.index[0].strftime('%Y-%m-%d'),
                                'end': period_data.index[-1].strftime('%Y-%m-%d'),
                                'data': period_data,
                                'days': len(period_data),
                                'price_start': period_data['close'].iloc[0],
                                'price_end': period_data['close'].iloc[-1],
                                'period_return': (period_data['close'].iloc[-1] / period_data['close'].iloc[0] - 1) * 100
                            })
        else:
            # Original logic for multi-year data
            train_end_date = f'{train_end_year}-12-31'
            train_data = df[df.index <= train_end_date].copy()
            
            # Define test periods (similar to AVAX but Bitcoin-specific)
            test_period_definitions = [
                ('Q1 2024', '2024-01-01', '2024-03-31'),
                ('Q2 2024', '2024-04-01', '2024-06-30'), 
                ('Q3 2024', '2024-07-01', '2024-09-30'),
                ('Q4 2024', '2024-10-01', '2024-12-31'),
                ('Q1 2025', '2025-01-01', '2025-03-31'),
                ('Q2 2025', '2025-04-01', '2025-06-30'),
                ('Current', '2025-07-01', datetime.now().strftime('%Y-%m-%d')),
            ]
            
            test_periods = []
            for name, start, end in test_period_definitions:
                period_data = df[(df.index >= start) & (df.index <= end)].copy()
                if len(period_data) > 10:  # Only include periods with sufficient data
                    test_periods.append({
                        'name': name,
                        'start': start,
                        'end': end, 
                        'data': period_data,
                        'days': len(period_data),
                        'price_start': period_data['close'].iloc[0],
                        'price_end': period_data['close'].iloc[-1],
                        'period_return': (period_data['close'].iloc[-1] / period_data['close'].iloc[0] - 1) * 100
                    })
        
        if len(train_data) == 0:
            raise ValueError("No training data available!")

        
        # Fixed the problematic print statement
        start_date_str = train_data.index[0].strftime('%Y-%m-%d')
        end_date_str = train_data.index[-1].strftime('%Y-%m-%d')
        train_days = len(train_data)
        num_periods = len(periods_data)
        total_start = df.index[0].strftime('%Y-%m-%d')
        total_end = df.index[-1].strftime('%Y-%m-%d')
        total_days = len(df)
        
        print(f"""
📊 BITCOIN TEST PERIODS CREATED:
   Training data: {start_date_str} to {end_date_str} ({train_days} days)
   Test periods: {num_periods} quarters/periods
   Total span: {total_start} to {total_end} ({total_days} days)""")
        
        for period in periods_data:
            print(f"   {period['name']}: {period['days']} days, {period['period_return']:+.1f}% period return")
        
        return train_data, periods_data
    
    def optimize_bitcoin_on_training_data(self, train_data, ngen=25, pop_size=50):
        """
        Optimize Bitcoin strategy parameters on training data only (no data leakage!)
        """
        # Fixed the problematic print statement
        start_date_str = train_data.index[0].strftime('%Y-%m-%d')
        end_date_str = train_data.index[-1].strftime('%Y-%m-%d')
        train_days = len(train_data)
        total_backtests = ngen * pop_size
        
        print(f"""
🎓 OPTIMIZING BITCOIN ON TRAINING DATA ONLY
===============================================
   Training period: {start_date_str} to {end_date_str}
   Training days: {train_days}
   Generations: {ngen}
   Population: {pop_size}
   Total backtests: ~{total_backtests:,}""")
        
        start_time = time.time()
        
        # Initialize DEAP (same as fixed optimizer)
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        fitness_config = FitnessConfig(
            profitability_weight=0.35,
            risk_adjusted_weight=0.35,
            drawdown_weight=0.20,
            trade_quality_weight=0.10,
            min_trades=2,
            min_win_rate=0.40,
            max_drawdown_threshold=0.25,
            min_profit_threshold=-0.08,
            min_fitness=0.01,
            no_trade_fitness=0.005
        )
        
        toolbox = base.Toolbox()
        
        def create_bitcoin_individual():
            return creator.Individual([
                random.uniform(1.0, 2.0),     # risk_reward
                random.uniform(1.0, 1.8),     # trend_strength
                random.uniform(0.5, 1.2),     # entry_threshold
                random.uniform(1.0, 1.8),     # confidence
                random.uniform(0.25, 0.70),   # buy_threshold_pct
                random.uniform(0.65, 0.85),   # sell_threshold_pct
                random.uniform(1.0, 1.2),     # bull_multiplier
                random.uniform(0.7, 0.9),     # bear_multiplier
                random.uniform(0.6, 0.8),     # high_vol_multiplier
                random.uniform(1.1, 1.4),     # low_vol_multiplier
                random.uniform(0.6, 1.5),     # max_position_pct
                random.uniform(0.01, 0.04),   # stop_loss_pct (POSITIVE ONLY)
                random.uniform(0.05, 0.12),   # take_profit_pct
            ])
        
        def constrain_individual(individual):
            individual[0] = max(0.5, min(3.0, individual[0]))    # risk_reward
            individual[1] = max(0.5, min(2.5, individual[1]))    # trend_strength  
            individual[2] = max(0.1, min(2.0, individual[2]))    # entry_threshold
            individual[3] = max(0.5, min(2.5, individual[3]))    # confidence
            individual[4] = max(0.20, min(0.80, individual[4]))  # buy_threshold_pct
            individual[5] = max(0.60, min(0.90, individual[5]))  # sell_threshold_pct
            
            # Ensure buy < sell
            if individual[4] >= individual[5] - 0.10:
                individual[4] = individual[5] - 0.15
                individual[4] = max(0.20, individual[4])
            
            individual[6] = max(0.9, min(1.3, individual[6]))    # bull_multiplier
            individual[7] = max(0.6, min(1.0, individual[7]))    # bear_multiplier
            individual[8] = max(0.5, min(0.9, individual[8]))    # high_vol_multiplier
            individual[9] = max(1.0, min(1.5, individual[9]))    # low_vol_multiplier
            individual[10] = max(0.4, min(2.0, individual[10]))  # max_position_pct
            individual[11] = max(0.005, min(0.05, abs(individual[11])))  # stop_loss_pct
            individual[12] = max(0.03, min(0.15, individual[12]))         # take_profit_pct
            
            # Ensure take_profit > stop_loss
            if individual[12] <= individual[11]:
                individual[12] = individual[11] + 0.02
                individual[12] = min(0.15, individual[12])
            
            return individual,
        
        def evaluate_individual(individual):
            try:
                constrain_individual(individual)
                
                params = BitcoinParams(
                    risk_reward=individual[0],
                    trend_strength=individual[1],
                    entry_threshold=individual[2],
                    confidence=individual[3],
                    buy_threshold_pct=individual[4],
                    sell_threshold_pct=individual[5],
                    bull_multiplier=individual[6],
                    bear_multiplier=individual[7],
                    high_vol_multiplier=individual[8],
                    low_vol_multiplier=individual[9],
                    max_position_pct=individual[10],
                    stop_loss_pct=individual[11],
                    take_profit_pct=individual[12]
                )
                
                results = evaluate_bitcoin_strategy_with_backtest(train_data, params, fitness_config)
                fitness = results.get('fitness', 0.01)
                
                return (max(0.01, min(1.0, fitness)),)
                
            except Exception as e:
                return (0.01,)
        
        toolbox.register("individual", create_bitcoin_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("constrain", constrain_individual)
        toolbox.register("evaluate", evaluate_individual)
        
        # Create population and optimize
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        
        print("🚀 Starting Bitcoin optimization on training data...")
        
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, 
            cxpb=0.6,
            mutpb=0.2,
            ngen=ngen,
            stats=stats, 
            halloffame=hof, 
            verbose=False
        )
        
        # Get best individual and create final parameters
        best = hof[0]
        toolbox.constrain(best)
        
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
        
        # Test on training data for reference
        train_results = evaluate_bitcoin_strategy_with_backtest(train_data, best_params, fitness_config)
        
        optimization_time = time.time() - start_time
        
        # Fixed the problematic print statement
        runtime_min = optimization_time / 60
        fitness_val = train_results['fitness']
        return_pct = train_results['returns']
        trades_num = train_results['trades']
        drawdown_pct = train_results['max_drawdown']
        buy_thresh = best_params.buy_threshold_pct
        sell_thresh = best_params.sell_threshold_pct
        stop_loss = best_params.stop_loss_pct
        take_profit = best_params.take_profit_pct
        
        print(f"""
🎉 TRAINING OPTIMIZATION COMPLETE!
=====================================
   Runtime: {runtime_min:.1f} minutes
   Best fitness: {fitness_val:.4f}
   Training return: {return_pct:.1%}
   Training trades: {trades_num}
   Training max drawdown: {drawdown_pct:.1%}
   
📊 Optimized Parameters:
   Buy threshold: {buy_thresh:.1%}
   Sell threshold: {sell_thresh:.1%}
   Stop loss: {stop_loss:.2%}
   Take profit: {take_profit:.1%}""")
        
        return best_params, train_results
    
    def test_bitcoin_on_periods(self, periods_data, optimized_params):
        """
        Test optimized parameters on each out-of-sample period
        """
        print("""
🧪 TESTING BITCOIN ON OUT-OF-SAMPLE PERIODS
============================================""")
        
        results = []
        fitness_config = FitnessConfig()
        
        for period in periods_data:
            print(f"\n📅 Testing {period['name']} ({period['start']} to {period['end']}, {period['days']} days)")
            print(f"   Period return: {period['period_return']:+.1f}%")
            
            try:
                # Run backtest on this period
                period_results = evaluate_bitcoin_strategy_with_backtest(
                    period['data'], optimized_params, fitness_config
                )
                
                result = {
                    'period': period['name'],
                    'start': period['start'],
                    'end': period['end'],
                    'days': period['days'],
                    'period_return': period['period_return'],
                    'strategy_return': period_results['returns'],
                    'trades': period_results['trades'],
                    'win_rate': period_results.get('win_rate', 0),
                    'max_drawdown': period_results['max_drawdown'],
                    'fitness': period_results['fitness'],
                    'status': 'success'
                }
                
                print(f"   Strategy return: {result['strategy_return']:8.1%}   vs   Period return: {result['period_return']:8.1%}")
                print(f"   Trades: {result['trades']}   Win rate: {result['win_rate']:.0%}   Drawdown: {result['max_drawdown']:6.1%}")
                
            except Exception as e:
                result = {
                    'period': period['name'],
                    'start': period['start'], 
                    'end': period['end'],
                    'days': period['days'],
                    'period_return': period['period_return'],
                    'strategy_return': 0.0,
                    'trades': 0,
                    'win_rate': 0.0,
                    'max_drawdown': 0.0,
                    'fitness': 0.0,
                    'status': f'error: {str(e)[:50]}'
                }
                
                print(f"   ❌ Error: {str(e)[:50]}")
            
            results.append(result)
        
        return results
    
    def analyze_bitcoin_validation_results(self, results, train_performance):
        """
        Comprehensive analysis of Bitcoin validation results
        """
        print("""
📊 BITCOIN VALIDATION ANALYSIS
==============================""")
        
        # Filter successful results
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            print("❌ No successful validation periods found!")
            return {'market_beat_rate': 0, 'profitable_rate': 0, 'expected_return': 0, 'max_drawdown': 0, 'avg_outperformance': 0}
        
        # Extract metrics
        strategy_returns = [r['strategy_return'] for r in successful_results]
        period_returns = [r['period_return'] / 100 for r in successful_results]  # Convert to decimal
        drawdowns = [r['max_drawdown'] for r in successful_results]
        trades_per_period = [r['trades'] for r in successful_results]
        
        # Calculate outperformance
        outperformance = [(s - p) * 100 for s, p in zip(strategy_returns, period_returns)]
        
        # Print results with individual variables
        successful_count = len(successful_results)
        total_count = len(results)
        profitable_count = len([r for r in strategy_returns if r > 0])
        strategy_win_rate = profitable_count / successful_count if successful_count > 0 else 0
        outperform_count = len([o for o in outperformance if o > 0])
        
        print(f"""
📈 VALIDATION RESULTS SUMMARY:
   Successful periods: {successful_count}/{total_count}
   Profitable periods: {profitable_count}/{successful_count}
   Strategy win rate: {strategy_win_rate:.1%}
   Outperformed market: {outperform_count}/{successful_count}""")
        
        # Strategy performance
        avg_return = np.mean(strategy_returns)
        median_return = np.median(strategy_returns)
        best_return = max(strategy_returns)
        worst_return = min(strategy_returns)
        std_return = np.std(strategy_returns)
        
        print(f"""
📊 STRATEGY PERFORMANCE:
   Average return: {avg_return:8.1%}
   Median return:  {median_return:8.1%}
   Best return:    {best_return:8.1%}
   Worst return:   {worst_return:8.1%}
   Std deviation:  {std_return:8.1%}""")
        
        # Market outperformance
        avg_outperf = np.mean(outperformance)
        median_outperf = np.median(outperformance)
        best_outperf = max(outperformance)
        worst_outperf = min(outperformance)
        
        print(f"""
📊 MARKET OUTPERFORMANCE:
   Average outperformance: {avg_outperf:+.1f}%
   Median outperformance:  {median_outperf:+.1f}%
   Best outperformance:    {best_outperf:+.1f}%
   Worst outperformance:   {worst_outperf:+.1f}%""")
        
        # Risk metrics
        avg_drawdown = np.mean(drawdowns)
        max_drawdown = max(drawdowns)
        avg_trades = np.mean(trades_per_period)
        
        print(f"""
📊 RISK METRICS:
   Average drawdown: {avg_drawdown:6.1%}
   Max drawdown:     {max_drawdown:6.1%}
   Average trades:   {avg_trades:.1f}""")
        
        # Detailed breakdown
        print("\n📅 DETAILED VALIDATION BREAKDOWN:")
        print(f"{'Period':<12} {'Strategy':<10} {'Market':<10} {'Outperf':<10} {'Trades':<7} {'Drawdown':<10}")
        print("-" * 70)
        
        for i, result in enumerate(successful_results):
            outperf = outperformance[i]
            print(f"{result['period']:<12} {result['strategy_return']:>8.1%} {result['period_return']:>8.1%} "
                  f"{outperf:>+8.1f}% {result['trades']:>5} {result['max_drawdown']:>8.1%}")
        
        # Comparison with training
        train_return = train_performance['returns']
        avg_validation = np.mean(strategy_returns)
        performance_ratio = avg_validation/train_return if train_return != 0 else 0
        degradation = (train_return - avg_validation)*100
        
        print(f"""
🔍 TRAINING vs VALIDATION:
   Training return:     {train_return:8.1%}
   Avg validation:      {avg_validation:8.1%}
   Performance ratio:   {performance_ratio:8.2f}x
   Degradation:         {degradation:+.1f}%""")
        
        # Final assessment
        market_beat_rate = len([o for o in outperformance if o > 0]) / len(outperformance)
        profitable_rate = len([r for r in strategy_returns if r > 0]) / len(strategy_returns)
        expected_return = np.median(strategy_returns)
        expected_drawdown = max(drawdowns)
        trade_frequency = np.mean(trades_per_period)
        
        print(f"""
💡 BITCOIN STRATEGY ASSESSMENT:
   Market beating rate: {market_beat_rate:.0%}
   Profitable rate:     {profitable_rate:.0%}
   Expected return:     {expected_return:8.1%} per period
   Expected drawdown:   {expected_drawdown:8.1%}
   Trade frequency:     {trade_frequency:.1f} trades per period""")
        
        return {
            'validation_returns': strategy_returns,
            'market_beat_rate': market_beat_rate,
            'profitable_rate': profitable_rate,
            'expected_return': expected_return,
            'max_drawdown': max_drawdown,
            'avg_outperformance': avg_outperf
        }

def main():
    """
    Main Bitcoin Rolling Validation workflow
    """
    print("₿ BITCOIN ROLLING VALIDATION - AVAX APPROACH")
    print("=" * 60)
    print("🎯 Extended timeframes for robust validation")
    print("📊 Train/test split with no data leakage")
    print("🔬 Multi-period out-of-sample testing")
    
    validator = BitcoinRollingValidator()
    
    try:
        # Step 1: Fetch extended Bitcoin data (3+ years like AVAX)
        print(f"\n{'='*20} STEP 1: DATA COLLECTION {'='*20}")
        bitcoin_df = validator.fetch_extended_bitcoin_data(days=1080)  # ~3 years
        
        # Step 2: Create proper train/test periods
        print(f"\n{'='*20} STEP 2: TRAIN/TEST SPLIT {'='*20}")
        train_data, test_periods = validator.create_bitcoin_test_periods(bitcoin_df, train_end_year=2023)
        
        # Step 3: Optimize on training data only
        print(f"\n{'='*20} STEP 3: TRAINING OPTIMIZATION {'='*20}")
        optimized_params, train_results = validator.optimize_bitcoin_on_training_data(
            train_data, ngen=30, pop_size=60  # More thorough optimization
        )
        
        # Step 4: Validate on out-of-sample periods
        print(f"\n{'='*20} STEP 4: OUT-OF-SAMPLE TESTING {'='*20}")
        validation_results = validator.test_bitcoin_on_periods(test_periods, optimized_params)
        
        # Step 5: Comprehensive analysis
        print(f"\n{'='*20} STEP 5: VALIDATION ANALYSIS {'='*20}")
        summary = validator.analyze_bitcoin_validation_results(validation_results, train_results)
        
        # Final results
        market_beat_rate = summary['market_beat_rate']
        profitable_rate = summary['profitable_rate']
        expected_return = summary['expected_return']
        max_drawdown = summary['max_drawdown']
        
        validation_status = 'PASSED ✅' if market_beat_rate > 0.5 and profitable_rate > 0.6 else 'NEEDS WORK ⚠️'
        
        print(f"""
🎉 BITCOIN ROLLING VALIDATION COMPLETE!
======================================
📊 Key Findings:
   Market beating rate: {market_beat_rate:.0%}
   Profitable periods:  {profitable_rate:.0%}
   Expected return:     {expected_return:.1%} per period
   Max drawdown:        {max_drawdown:.1%}
   
💡 Validation Status: {validation_status}""")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"bitcoin_rolling_validation_{timestamp}.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump({
                'validation_summary': summary,
                'train_results': train_results,
                'validation_results': validation_results,
                'optimized_params': optimized_params.__dict__,
                'timestamp': timestamp
            }, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Bitcoin rolling validation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()