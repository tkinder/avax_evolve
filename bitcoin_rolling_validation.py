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
        Using CoinGecko for longer historical data
        """
        print(f"📊 Fetching {days} days of Bitcoin historical data...")
        
        # CoinGecko allows up to 365 days per request for daily data
        # For longer periods, we need multiple requests
        all_data = []
        current_date = datetime.now()
        
        # Split into chunks of 365 days
        chunks = (days // 365) + 1
        
        for i in range(chunks):
            end_date = current_date - timedelta(days=i * 365)
            start_date = end_date - timedelta(days=min(365, days - i * 365))
            
            if days - i * 365 <= 0:
                break
                
            print(f"   Fetching chunk {i+1}/{chunks}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # CoinGecko API call
            url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': int(start_date.timestamp()),
                'to': int(end_date.timestamp())
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Extract price data
                prices = data['prices']
                volumes = data['total_volumes']
                
                # Convert to DataFrame
                df_chunk = pd.DataFrame(prices, columns=['timestamp', 'close'])
                df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
                df_chunk.set_index('timestamp', inplace=True)
                
                # Add volume data
                volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                volume_df.set_index('timestamp', inplace=True)
                df_chunk = df_chunk.join(volume_df, how='left')
                
                # Add OHLC approximations (CoinGecko only gives close prices for long ranges)
                df_chunk['open'] = df_chunk['close'].shift(1)
                df_chunk['high'] = df_chunk[['open', 'close']].max(axis=1)
                df_chunk['low'] = df_chunk[['open', 'close']].min(axis=1)
                
                all_data.append(df_chunk)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"   ❌ Error fetching chunk {i+1}: {e}")
                continue
        
        if not all_data:
            raise ValueError("Failed to fetch any Bitcoin data")
        
        # Combine all chunks
        df = pd.concat(all_data)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
        df = df.dropna()
        
        print(f"✅ Bitcoin data ready: {len(df)} days from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
        print(f"   Current price: ${df['close'].iloc[-1]:,.0f}")
        
        return df
    
    def create_bitcoin_test_periods(self, df, train_end_year=2023):
        """
        Create test periods similar to AVAX but adapted for Bitcoin
        Train on data up to train_end_year, test on subsequent periods
        """
        train_end_date = f'{train_end_year}-12-31'
        train_data = df[df.index <= train_end_date].copy()
        
        # Define test periods (similar to AVAX but Bitcoin-specific)
        test_periods = [
            ('Q1 2024', '2024-01-01', '2024-03-31'),
            ('Q2 2024', '2024-04-01', '2024-06-30'), 
            ('Q3 2024', '2024-07-01', '2024-09-30'),
            ('Q4 2024', '2024-10-01', '2024-12-31'),
            ('Q1 2025', '2025-01-01', '2025-03-31'),
            ('Q2 2025', '2025-04-01', '2025-06-30'),
            ('Current', '2025-07-01', datetime.now().strftime('%Y-%m-%d')),
        ]
        
        periods_data = []
        for name, start, end in test_periods:
            period_data = df[(df.index >= start) & (df.index <= end)].copy()
            if len(period_data) > 10:  # Only include periods with sufficient data
                periods_data.append({
                    'name': name,
                    'start': start,
                    'end': end, 
                    'data': period_data,
                    'days': len(period_data),
                    'price_start': period_data['close'].iloc[0],
                    'price_end': period_data['close'].iloc[-1],
                    'period_return': (period_data['close'].iloc[-1] / period_data['close'].iloc[0] - 1) * 100
                })
        
        print(f\"\"\"
📊 BITCOIN TEST PERIODS CREATED:
   Training data: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')} ({len(train_data)} days)
   Test periods: {len(periods_data)} quarters/periods
   Total span: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} days)\"\"\")
        
        for period in periods_data:
            print(f\"   {period['name']}: {period['days']} days, {period['period_return']:+.1f}% period return\")
        
        return train_data, periods_data
    
    def optimize_bitcoin_on_training_data(self, train_data, ngen=25, pop_size=50):
        """
        Optimize Bitcoin strategy parameters on training data only (no data leakage!)
        """
        print(f\"\"\"
🎓 OPTIMIZING BITCOIN ON TRAINING DATA ONLY
===============================================
   Training period: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}
   Training days: {len(train_data)}
   Generations: {ngen}
   Population: {pop_size}
   Total backtests: ~{ngen * pop_size:,}\"\"\")
        
        start_time = time.time()
        
        # Initialize DEAP (same as fixed optimizer)
        if not hasattr(creator, \"FitnessMax\"):
            creator.create(\"FitnessMax\", base.Fitness, weights=(1.0,))
        if not hasattr(creator, \"Individual\"):
            creator.create(\"Individual\", list, fitness=creator.FitnessMax)
        
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
        
        toolbox.register(\"individual\", create_bitcoin_individual)
        toolbox.register(\"population\", tools.initRepeat, list, toolbox.individual)
        toolbox.register(\"mate\", tools.cxTwoPoint)
        toolbox.register(\"mutate\", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
        toolbox.register(\"select\", tools.selTournament, tournsize=3)
        toolbox.register(\"constrain\", constrain_individual)
        toolbox.register(\"evaluate\", evaluate_individual)
        
        # Create population and optimize
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register(\"avg\", np.mean)
        stats.register(\"max\", np.max)
        
        print(f\"🚀 Starting Bitcoin optimization on training data...\")
        
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
        
        print(f\"\"\"
🎉 TRAINING OPTIMIZATION COMPLETE!
=====================================
   Runtime: {optimization_time / 60:.1f} minutes
   Best fitness: {train_results['fitness']:.4f}
   Training return: {train_results['returns']:.1%}
   Training trades: {train_results['trades']}
   Training max drawdown: {train_results['max_drawdown']:.1%}
   
📊 Optimized Parameters:
   Buy threshold: {best_params.buy_threshold_pct:.1%}
   Sell threshold: {best_params.sell_threshold_pct:.1%}
   Stop loss: {best_params.stop_loss_pct:.2%}
   Take profit: {best_params.take_profit_pct:.1%}\"\"\")
        
        return best_params, train_results
    
    def test_bitcoin_on_periods(self, periods_data, optimized_params):
        """
        Test optimized parameters on each out-of-sample period
        """
        print(f\"\"\"
🧪 TESTING BITCOIN ON OUT-OF-SAMPLE PERIODS
===========================================\"\"\")
        
        results = []
        fitness_config = FitnessConfig()
        
        for period in periods_data:
            print(f\"\\n📅 Testing {period['name']} ({period['start']} to {period['end']}, {period['days']} days)\")
            print(f\"   Period return: {period['period_return']:+.1f}%\")
            
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
                
                print(f\"   Strategy return: {result['strategy_return']:8.1%}   vs   Period return: {result['period_return']:8.1%}\")
                print(f\"   Trades: {result['trades']}   Win rate: {result['win_rate']:.0%}   Drawdown: {result['max_drawdown']:6.1%}\")
                
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
                
                print(f\"   ❌ Error: {str(e)[:50]}\")
            
            results.append(result)
        
        return results
    
    def analyze_bitcoin_validation_results(self, results, train_performance):
        """
        Comprehensive analysis of Bitcoin validation results
        """
        print(f\"\"\"
📊 BITCOIN VALIDATION ANALYSIS
=============================\"\"\")
        
        # Filter successful results
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            print(\"❌ No successful validation periods found!\")
            return
        
        # Extract metrics
        strategy_returns = [r['strategy_return'] for r in successful_results]
        period_returns = [r['period_return'] / 100 for r in successful_results]  # Convert to decimal
        drawdowns = [r['max_drawdown'] for r in successful_results]
        trades_per_period = [r['trades'] for r in successful_results]
        
        # Calculate outperformance
        outperformance = [(s - p) * 100 for s, p in zip(strategy_returns, period_returns)]
        
        print(f\"\"\"
📈 VALIDATION RESULTS SUMMARY:
   Successful periods: {len(successful_results)}/{len(results)}
   Profitable periods: {len([r for r in strategy_returns if r > 0])}/{len(successful_results)}
   Strategy win rate: {len([r for r in strategy_returns if r > 0]) / len(successful_results):.1%}
   Outperformed market: {len([o for o in outperformance if o > 0])}/{len(successful_results)}\"\"\")
        
        print(f\"\"\"
📊 STRATEGY PERFORMANCE:
   Average return: {np.mean(strategy_returns):8.1%}
   Median return:  {np.median(strategy_returns):8.1%}
   Best return:    {max(strategy_returns):8.1%}
   Worst return:   {min(strategy_returns):8.1%}
   Std deviation:  {np.std(strategy_returns):8.1%}\"\"\")
        
        print(f\"\"\"
📊 MARKET OUTPERFORMANCE:
   Average outperformance: {np.mean(outperformance):+.1f}%
   Median outperformance:  {np.median(outperformance):+.1f}%
   Best outperformance:    {max(outperformance):+.1f}%
   Worst outperformance:   {min(outperformance):+.1f}%\"\"\")
        
        print(f\"\"\"
📊 RISK METRICS:
   Average drawdown: {np.mean(drawdowns):6.1%}
   Max drawdown:     {max(drawdowns):6.1%}
   Average trades:   {np.mean(trades_per_period):.1f}\"\"\")
        
        # Detailed breakdown
        print(f\"\\n📅 DETAILED VALIDATION BREAKDOWN:\")
        print(f\"{'Period':<12} {'Strategy':<10} {'Market':<10} {'Outperf':<10} {'Trades':<7} {'Drawdown':<10}\")
        print(\"-\" * 70)
        
        for i, result in enumerate(successful_results):
            outperf = outperformance[i]
            print(f\"{result['period']:<12} {result['strategy_return']:>8.1%} {result['period_return']:>8.1%} \"
                  f\"{outperf:>+8.1f}% {result['trades']:>5} {result['max_drawdown']:>8.1%}\")
        
        # Comparison with training
        train_return = train_performance['returns']
        avg_validation = np.mean(strategy_returns)
        
        print(f\"\"\"
🔍 TRAINING vs VALIDATION:
   Training return:     {train_return:8.1%}
   Avg validation:      {avg_validation:8.1%}
   Performance ratio:   {avg_validation/train_return if train_return != 0 else 0:8.2f}x
   Degradation:         {(train_return - avg_validation)*100:+.1f}%\"\"\")
        
        # Final assessment
        market_beat_rate = len([o for o in outperformance if o > 0]) / len(outperformance)
        profitable_rate = len([r for r in strategy_returns if r > 0]) / len(strategy_returns)
        
        print(f\"\"\"
💡 BITCOIN STRATEGY ASSESSMENT:
   Market beating rate: {market_beat_rate:.0%}
   Profitable rate:     {profitable_rate:.0%}
   Expected return:     {np.median(strategy_returns):8.1%} per period
   Expected drawdown:   {max(drawdowns):8.1%}
   Trade frequency:     {np.mean(trades_per_period):.1f} trades per period\"\"\")
        
        return {
            'validation_returns': strategy_returns,
            'market_beat_rate': market_beat_rate,
            'profitable_rate': profitable_rate,
            'expected_return': np.median(strategy_returns),
            'max_drawdown': max(drawdowns),
            'avg_outperformance': np.mean(outperformance)
        }

def main():
    \"\"\"
    Main Bitcoin Rolling Validation workflow
    \"\"\"
    print(\"₿ BITCOIN ROLLING VALIDATION - AVAX APPROACH\")
    print(\"=\" * 60)
    print(\"🎯 Extended timeframes for robust validation\")
    print(\"📊 Train/test split with no data leakage\")
    print(\"🔬 Multi-period out-of-sample testing\")
    
    validator = BitcoinRollingValidator()
    
    try:
        # Step 1: Fetch extended Bitcoin data (3+ years like AVAX)
        print(f\"\\n{'='*20} STEP 1: DATA COLLECTION {'='*20}\")
        bitcoin_df = validator.fetch_extended_bitcoin_data(days=1080)  # ~3 years
        
        # Step 2: Create proper train/test periods
        print(f\"\\n{'='*20} STEP 2: TRAIN/TEST SPLIT {'='*20}\")
        train_data, test_periods = validator.create_bitcoin_test_periods(bitcoin_df, train_end_year=2023)
        
        # Step 3: Optimize on training data only
        print(f\"\\n{'='*20} STEP 3: TRAINING OPTIMIZATION {'='*20}\")
        optimized_params, train_results = validator.optimize_bitcoin_on_training_data(
            train_data, ngen=30, pop_size=60  # More thorough optimization
        )
        
        # Step 4: Validate on out-of-sample periods
        print(f\"\\n{'='*20} STEP 4: OUT-OF-SAMPLE TESTING {'='*20}\")
        validation_results = validator.test_bitcoin_on_periods(test_periods, optimized_params)
        
        # Step 5: Comprehensive analysis
        print(f\"\\n{'='*20} STEP 5: VALIDATION ANALYSIS {'='*20}\")
        summary = validator.analyze_bitcoin_validation_results(validation_results, train_results)
        
        print(f\"\"\"
🎉 BITCOIN ROLLING VALIDATION COMPLETE!
======================================
📊 Key Findings:
   Market beating rate: {summary['market_beat_rate']:.0%}
   Profitable periods:  {summary['profitable_rate']:.0%}
   Expected return:     {summary['expected_return']:.1%} per period
   Max drawdown:        {summary['max_drawdown']:.1%}
   
💡 Validation Status: {'PASSED ✅' if summary['market_beat_rate'] > 0.5 and summary['profitable_rate'] > 0.6 else 'NEEDS WORK ⚠️'}
\"\"\")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f\"bitcoin_rolling_validation_{timestamp}.json\"
        
        import json
        with open(results_file, 'w') as f:
            json.dump({
                'validation_summary': summary,
                'train_results': train_results,
                'validation_results': validation_results,
                'optimized_params': optimized_params.__dict__,
                'timestamp': timestamp
            }, f, indent=2, default=str)
        
        print(f\"💾 Results saved to: {results_file}\")
        
    except Exception as e:
        print(f\"❌ Bitcoin rolling validation error: {e}\")
        import traceback
        traceback.print_exc()

if __name__ == \"__main__\":
    main()
