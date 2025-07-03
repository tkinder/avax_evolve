#!/usr/bin/env python3
"""
Solana Complete Validation - Full AVAX-Style Approach
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

# Import Solana components
from solana_backtester import SolanaParams, evaluate_solana_strategy_with_backtest
from src.evolution.fitness import FitnessConfig

# Import DEAP for optimization
from deap import base, creator, tools, algorithms
import random

class SolanaCompleteValidator:
    def __init__(self):
        print("🌟 Solana Complete Validation System")
        print("🎯 Full AVAX-style train/test validation")
        
    def fetch_solana_data(self, days=1080):
        """Fetch Solana data using CryptoCompare API (same as AVAX/Bitcoin)"""
        print(f"📊 Fetching {days} days of Solana historical data...")
        
        API_KEY = '4c322523f98c6c20dbe789194197dafac7329ec5a7dc378503118d443e867c2b'
        
        end_date = pd.Timestamp.today() - pd.Timedelta(days=1)
        url = (
            f"https://min-api.cryptocompare.com/data/v2/histoday?"
            f"fsym=SOL&tsym=USD&limit={days}&toTs={int(end_date.timestamp())}&api_key={API_KEY}"
        )
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()['Data']['Data']
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            df = df.rename(columns={
                'close': 'close', 'high': 'high', 'low': 'low', 'open': 'open',
                'volumefrom': 'volume', 'volumeto': 'volumeto'
            })
            
            numeric_columns = ['close', 'high', 'low', 'open', 'volume', 'volumeto']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            print(f"✅ Solana data ready: {len(df)} days ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching Solana data: {e}")
            raise
    
    def create_train_test_split(self, df):
        """Create train/test split based on available data"""
        # Use year-based split: train on pre-2024, test on 2024+
        train_data = df[df.index < '2024-01-01'].copy()
        test_data = df[df.index >= '2024-01-01'].copy()
        
        # Split test data into 4 periods
        test_periods = []
        chunk_size = len(test_data) // 4
        
        for i in range(4):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 3 else len(test_data)
            
            period_data = test_data.iloc[start_idx:end_idx]
            test_periods.append({
                'name': f'Test_P{i+1}',
                'start': period_data.index[0].strftime('%Y-%m-%d'),
                'end': period_data.index[-1].strftime('%Y-%m-%d'),
                'data': period_data,
                'days': len(period_data),
                'period_return': (period_data['close'].iloc[-1] / period_data['close'].iloc[0] - 1) * 100
            })
        
        print(f"📊 Split: {len(train_data)} train days, {len(test_periods)} test periods")
        return train_data, test_periods
    
    def optimize_on_training_data(self, train_data, ngen=25, pop_size=50):
        """Optimize Solana strategy on training data only"""
        print(f"\n🎓 OPTIMIZING ON TRAINING DATA ({len(train_data)} days)")
        print(f"   Generations: {ngen}, Population: {pop_size}")
        
        start_time = time.time()
        
        # Initialize DEAP
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Solana-specific fitness config (balanced approach)
        fitness_config = FitnessConfig(
            profitability_weight=0.40,     # Focus on profits
            risk_adjusted_weight=0.25,     # Risk-adjusted returns
            drawdown_weight=0.25,          # Drawdown control
            trade_quality_weight=0.10,     # Trade quality
            min_trades=2,                  # Solana contrarian - fewer trades ok
            min_win_rate=0.35,             # Lower win rate tolerance
            max_drawdown_threshold=0.45,   # Higher drawdown tolerance
            min_profit_threshold=-0.15     # Allow moderate losses
        )
        
        toolbox = base.Toolbox()
        
        def create_individual():
            return creator.Individual([
                random.uniform(1.0, 2.0),     # risk_reward
                random.uniform(1.0, 1.8),     # trend_strength
                random.uniform(0.5, 1.2),     # entry_threshold
                random.uniform(1.0, 1.8),     # confidence
                random.uniform(0.20, 0.60),   # buy_threshold_pct (Solana range)
                random.uniform(0.55, 0.85),   # sell_threshold_pct (Solana range)
                random.uniform(0.9, 1.3),     # bull_multiplier
                random.uniform(0.6, 0.9),     # bear_multiplier
                random.uniform(0.7, 1.0),     # high_vol_multiplier
                random.uniform(1.0, 1.3),     # low_vol_multiplier
                random.uniform(0.4, 1.0),     # max_position_pct (conservative)
                random.uniform(0.10, 0.45),   # stop_loss_pct (Solana high tolerance)
                random.uniform(0.08, 0.30),   # take_profit_pct (Solana range)
            ])
        
        def constrain_individual(individual):
            individual[0] = max(0.5, min(3.0, individual[0]))    # risk_reward
            individual[1] = max(0.5, min(2.5, individual[1]))    # trend_strength  
            individual[2] = max(0.1, min(2.0, individual[2]))    # entry_threshold
            individual[3] = max(0.5, min(2.5, individual[3]))    # confidence
            individual[4] = max(0.15, min(0.70, individual[4]))  # buy_threshold_pct
            individual[5] = max(0.50, min(0.90, individual[5]))  # sell_threshold_pct
            
            # Ensure buy < sell
            if individual[4] >= individual[5] - 0.10:
                individual[4] = individual[5] - 0.15
                individual[4] = max(0.15, individual[4])
            
            individual[6] = max(0.8, min(1.4, individual[6]))    # bull_multiplier
            individual[7] = max(0.5, min(1.0, individual[7]))    # bear_multiplier
            individual[8] = max(0.6, min(1.1, individual[8]))    # high_vol_multiplier
            individual[9] = max(0.9, min(1.4, individual[9]))    # low_vol_multiplier
            individual[10] = max(0.3, min(1.2, individual[10]))  # max_position_pct
            individual[11] = max(0.05, min(0.50, abs(individual[11])))  # stop_loss_pct
            individual[12] = max(0.05, min(0.35, individual[12]))         # take_profit_pct
            
            # Ensure take_profit > stop_loss (unless stop loss is very high - Solana contrarian)
            if individual[11] < 0.20 and individual[12] <= individual[11]:
                individual[12] = individual[11] + 0.03
            
            return individual,
        
        def evaluate_individual(individual):
            try:
                constrain_individual(individual)
                
                params = SolanaParams(
                    risk_reward=individual[0], trend_strength=individual[1],
                    entry_threshold=individual[2], confidence=individual[3],
                    buy_threshold_pct=individual[4], sell_threshold_pct=individual[5],
                    bull_multiplier=individual[6], bear_multiplier=individual[7],
                    high_vol_multiplier=individual[8], low_vol_multiplier=individual[9],
                    max_position_pct=individual[10], stop_loss_pct=individual[11],
                    take_profit_pct=individual[12]
                )
                
                results = evaluate_solana_strategy_with_backtest(train_data, params, fitness_config)
                return (max(0.01, min(1.0, results.get('fitness', 0.01))),)
                
            except Exception:
                return (0.01,)
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate_individual)
        
        # Run optimization
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        
        print("🚀 Starting optimization...")
        pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.2, ngen=ngen, 
                                   halloffame=hof, verbose=False)
        
        # Get best parameters
        best = hof[0]
        constrain_individual(best)
        
        best_params = SolanaParams(
            risk_reward=best[0], trend_strength=best[1], entry_threshold=best[2],
            confidence=best[3], buy_threshold_pct=best[4], sell_threshold_pct=best[5],
            bull_multiplier=best[6], bear_multiplier=best[7], 
            high_vol_multiplier=best[8], low_vol_multiplier=best[9],
            max_position_pct=best[10], stop_loss_pct=best[11], take_profit_pct=best[12]
        )
        
        # Test on training data
        train_results = evaluate_solana_strategy_with_backtest(train_data, best_params, fitness_config)
        
        runtime = time.time() - start_time
        print(f"✅ Optimization complete! ({runtime/60:.1f} minutes)")
        print(f"   Training performance: {train_results['returns']:.1%} return, {train_results['trades']} trades")
        
        return best_params, train_results
    
    def test_on_periods(self, test_periods, params):
        """Test optimized parameters on each out-of-sample period"""
        print(f"\n🧪 TESTING ON OUT-OF-SAMPLE PERIODS")
        
        results = []
        fitness_config = FitnessConfig()
        
        for period in test_periods:
            print(f"\n📅 {period['name']}: {period['days']} days, market return {period['period_return']:+.1f}%")
            
            try:
                test_results = evaluate_solana_strategy_with_backtest(
                    period['data'], params, fitness_config
                )
                
                result = {
                    'period': period['name'],
                    'days': period['days'],
                    'market_return': period['period_return'],
                    'strategy_return': test_results['returns'] * 100,  # Convert to %
                    'trades': test_results['trades'],
                    'win_rate': test_results.get('win_rate', 0) * 100,  # Convert to %
                    'max_drawdown': test_results['max_drawdown'] * 100,  # Convert to %
                    'outperformance': (test_results['returns'] * 100) - period['period_return'],
                    'status': 'success'
                }
                
                print(f"   Strategy: {result['strategy_return']:+.1f}% vs Market: {result['market_return']:+.1f}% = {result['outperformance']:+.1f}% outperformance")
                print(f"   Trades: {result['trades']}, Win rate: {result['win_rate']:.0f}%, Drawdown: {result['max_drawdown']:.1f}%")
                
            except Exception as e:
                result = {
                    'period': period['name'], 'days': period['days'],
                    'market_return': period['period_return'], 'strategy_return': 0,
                    'trades': 0, 'win_rate': 0, 'max_drawdown': 0,
                    'outperformance': -period['period_return'],
                    'status': f'error: {str(e)[:30]}'
                }
                print(f"   ❌ Error: {str(e)[:50]}")
            
            results.append(result)
        
        return results
    
    def analyze_results(self, results, train_results):
        """Analyze validation results"""
        print(f"\n📊 VALIDATION ANALYSIS")
        print("=" * 50)
        
        successful = [r for r in results if r['status'] == 'success']
        
        if not successful:
            print("❌ No successful test periods!")
            return {'validation_status': 'FAILED'}
        
        strategy_returns = [r['strategy_return'] for r in successful]
        market_returns = [r['market_return'] for r in successful]
        outperformances = [r['outperformance'] for r in successful]
        
        # Summary stats
        profitable_periods = len([r for r in strategy_returns if r > 0])
        beat_market_periods = len([r for r in outperformances if r > 0])
        
        print(f"📈 SUMMARY:")
        print(f"   Successful periods: {len(successful)}/{len(results)}")
        print(f"   Profitable periods: {profitable_periods}/{len(successful)} ({profitable_periods/len(successful)*100:.0f}%)")
        print(f"   Beat market periods: {beat_market_periods}/{len(successful)} ({beat_market_periods/len(successful)*100:.0f}%)")
        
        print(f"\n📊 STRATEGY PERFORMANCE:")
        print(f"   Average return: {np.mean(strategy_returns):+.1f}%")
        print(f"   Best return: {max(strategy_returns):+.1f}%")
        print(f"   Worst return: {min(strategy_returns):+.1f}%")
        
        print(f"\n📊 MARKET OUTPERFORMANCE:")
        print(f"   Average outperformance: {np.mean(outperformances):+.1f}%")
        print(f"   Best outperformance: {max(outperformances):+.1f}%")
        print(f"   Worst outperformance: {min(outperformances):+.1f}%")
        
        # Detailed breakdown
        print(f"\n📅 DETAILED BREAKDOWN:")
        print(f"{'Period':<8} {'Strategy':<10} {'Market':<10} {'Outperf':<10} {'Trades':<7}")
        print("-" * 50)
        for r in successful:
            print(f"{r['period']:<8} {r['strategy_return']:>+8.1f}% {r['market_return']:>+8.1f}% "
                  f"{r['outperformance']:>+8.1f}% {r['trades']:>5}")
        
        # Training vs validation
        train_return = train_results['returns'] * 100
        avg_test_return = np.mean(strategy_returns)
        
        print(f"\n🔍 TRAINING vs VALIDATION:")
        print(f"   Training return: {train_return:+.1f}%")
        print(f"   Average test return: {avg_test_return:+.1f}%")
        print(f"   Performance degradation: {train_return - avg_test_return:+.1f}%")
        
        # Final verdict
        market_beat_rate = beat_market_periods / len(successful)
        profit_rate = profitable_periods / len(successful)
        
        # Solana-specific criteria (more lenient than Bitcoin)
        if market_beat_rate >= 0.5 and profit_rate >= 0.75:
            status = "EXCELLENT ✅"
        elif market_beat_rate >= 0.25 and profit_rate >= 0.5:
            status = "GOOD ✅"
        elif profit_rate >= 0.5:
            status = "ACCEPTABLE ⚠️"
        else:
            status = "NEEDS WORK ❌"
        
        print(f"\n💡 VALIDATION STATUS: {status}")
        print(f"   Expected return: {np.median(strategy_returns):+.1f}% per period")
        print(f"   Market beating rate: {market_beat_rate:.0%}")
        
        return {
            'validation_status': status,
            'market_beat_rate': market_beat_rate,
            'profit_rate': profit_rate,
            'expected_return': np.median(strategy_returns),
            'avg_outperformance': np.mean(outperformances)
        }

def main():
    """Main validation workflow"""
    print("🌟 SOLANA COMPLETE VALIDATION")
    print("=" * 60)
    print("🎯 Full train/test validation like AVAX & Bitcoin")
    
    validator = SolanaCompleteValidator()
    
    try:
        # Step 1: Get data
        print(f"\n{'='*15} STEP 1: DATA COLLECTION {'='*15}")
        df = validator.fetch_solana_data(days=1080)
        
        # Step 2: Split data
        print(f"\n{'='*15} STEP 2: TRAIN/TEST SPLIT {'='*15}")
        train_data, test_periods = validator.create_train_test_split(df)
        
        # Step 3: Optimize on training data
        print(f"\n{'='*15} STEP 3: OPTIMIZATION {'='*15}")
        best_params, train_results = validator.optimize_on_training_data(
            train_data, ngen=25, pop_size=50
        )
        
        # Step 4: Test on validation periods
        print(f"\n{'='*15} STEP 4: VALIDATION {'='*15}")
        test_results = validator.test_on_periods(test_periods, best_params)
        
        # Step 5: Analysis
        print(f"\n{'='*15} STEP 5: ANALYSIS {'='*15}")
        summary = validator.analyze_results(test_results, train_results)
        
        # Final result
        print(f"\n🎉 SOLANA VALIDATION COMPLETE!")
        print(f"💡 The enhanced Solana optimizer result is: {summary['validation_status']}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"solana_complete_validation_{timestamp}.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'train_results': train_results,
                'test_results': test_results,
                'best_params': best_params.__dict__,
                'timestamp': timestamp
            }, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
        
        # Comparison to enhanced results
        print(f"\n🔍 OVERFITTING CHECK:")
        print(f"   Enhanced optimizer result: +0.1% return")
        print(f"   Validation average result: {summary['expected_return']:+.1f}% return")
        if abs(0.1 - summary['expected_return']) < 5.0:  # Within 5% difference
            print(f"   ✅ Results consistent - not overfitted")
        else:
            print(f"   ⚠️  Significant difference - possible overfitting")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
