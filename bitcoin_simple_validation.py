#!/usr/bin/env python3
"""
Bitcoin Simple Validation - Using CryptoCompare API
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

class BitcoinValidator:
    def __init__(self):
        print("₿ Bitcoin Validation System Initialized")
        print("🎯 Using CryptoCompare API like AVAX")
        
    def fetch_bitcoin_data(self, days=1080):
        """Fetch Bitcoin data using CryptoCompare API (same as AVAX)"""
        print(f"📊 Fetching {days} days of Bitcoin historical data...")
        
        API_KEY = '4c322523f98c6c20dbe789194197dafac7329ec5a7dc378503118d443e867c2b'
        
        end_date = pd.Timestamp.today() - pd.Timedelta(days=1)
        url = (
            f"https://min-api.cryptocompare.com/data/v2/histoday?"
            f"fsym=BTC&tsym=USD&limit={days}&toTs={int(end_date.timestamp())}&api_key={API_KEY}"
        )
        
        try:
            print("   Fetching Bitcoin data from CryptoCompare...")
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
            print(f"❌ Error fetching Bitcoin data: {e}")
            raise
    
    def create_train_test_split(self, df):
        """Create train/test split based on available data"""
        earliest_date = df.index[0]
        latest_date = df.index[-1]
        
        print(f"   Data range: {earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
        
        # If we have multi-year data, use year-based split
        if earliest_date.year < 2024:
            print("   Using year-based split: train on pre-2024, test on 2024+")
            train_data = df[df.index < '2024-01-01'].copy()
            test_data = df[df.index >= '2024-01-01'].copy()
        else:
            # Use 60/40 split on available data
            print("   Using 60/40 split: first 60% training, last 40% testing")
            total_days = len(df)
            train_days = int(total_days * 0.6)
            train_data = df.iloc[:train_days].copy()
            test_data = df.iloc[train_days:].copy()
        
        if len(train_data) < 30:
            raise ValueError("Insufficient training data")
        if len(test_data) < 10:
            raise ValueError("Insufficient test data")
            
        # Split test data into 4 periods
        test_periods = []
        chunk_size = len(test_data) // 4
        
        for i in range(4):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 3 else len(test_data)
            
            if start_idx < len(test_data):
                period_data = test_data.iloc[start_idx:end_idx]
                if len(period_data) > 5:
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
        
        print(f"""
📊 TRAIN/TEST SPLIT CREATED:
   Training: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')} ({len(train_data)} days)
   Testing: {len(test_periods)} periods ({len(test_data)} total days)""")
        
        for period in test_periods:
            print(f"   {period['name']}: {period['days']} days, {period['period_return']:+.1f}% return")
        
        return train_data, test_periods

def main():
    """Main validation workflow"""
    print("₿ BITCOIN VALIDATION - CRYPTOCOMPARE APPROACH")
    print("=" * 60)
    
    validator = BitcoinValidator()
    
    try:
        # Step 1: Fetch Bitcoin data
        print(f"\n{'='*20} STEP 1: DATA COLLECTION {'='*20}")
        bitcoin_df = validator.fetch_bitcoin_data(days=1080)
        
        # Step 2: Create train/test split
        print(f"\n{'='*20} STEP 2: TRAIN/TEST SPLIT {'='*20}")
        train_data, test_periods = validator.create_train_test_split(bitcoin_df)
        
        print(f"\n🎉 SUCCESS! Got {len(train_data)} training days and {len(test_periods)} test periods")
        print("Ready for optimization and validation!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
