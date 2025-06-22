# hybrid_data_pipeline.py
import ccxt
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

load_dotenv()

class HybridDataPipeline:
    """
    Combines external historical data with live Binance testnet data
    """
    
    def __init__(self):
        # Initialize Binance testnet for live data
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_TESTNET_API_KEY'),
            'secret': os.getenv('BINANCE_TESTNET_SECRET'),
            'sandbox': True,
            'enableRateLimit': True,
        })
        
        print("🔄 Hybrid Data Pipeline Initialized")
        print("📊 External data: CoinGecko (historical)")
        print("🎯 Live data: Binance Testnet (current)")
        
    def fetch_historical_data_coingecko(self, symbol='avalanche-2', days=100):
        """
        Fetch historical AVAX data from CoinGecko (free API)
        """
        try:
            print(f"📥 Fetching {days} days of historical data from CoinGecko...")
            
            # CoinGecko API endpoint
            url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Create OHLC data (CoinGecko only gives daily close prices)
            # For backtesting, we'll approximate OHLC from close prices
            df['open'] = df['close'].shift(1)
            df['high'] = df[['open', 'close']].max(axis=1) * 1.02  # Approximate high
            df['low'] = df[['open', 'close']].min(axis=1) * 0.98   # Approximate low
            df['volume'] = 1000000  # Placeholder volume
            
            # Remove first row (no open price)
            df = df.dropna()
            
            # Reorder columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            print(f"✅ Got {len(df)} days from CoinGecko")
            print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ CoinGecko data fetch failed: {e}")
            return None
    
    def fetch_live_data_binance(self, symbol='AVAX/USDT', limit=20):
        """
        Fetch recent data from Binance testnet
        """
        try:
            print(f"🎯 Fetching live data from Binance testnet...")
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1d', limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ Got {len(df)} days from Binance testnet")
            print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Current price: ${df['close'].iloc[-1]:.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Binance live data fetch failed: {e}")
            return None
    
    def merge_historical_and_live(self, historical_df, live_df):
        """
        Merge historical and live data, avoiding duplicates
        """
        try:
            print("🔄 Merging historical and live data...")
            
            if historical_df is None or live_df is None:
                print("❌ Cannot merge - missing data")
                return None
            
            # Find overlap and remove duplicates
            live_start = live_df.index[0]
            
            # Keep historical data up to live data start
            historical_clean = historical_df[historical_df.index < live_start]
            
            # Combine
            merged_df = pd.concat([historical_clean, live_df])
            merged_df = merged_df.sort_index()
            
            print(f"✅ Merged data: {len(merged_df)} total days")
            print(f"   Historical: {len(historical_clean)} days")
            print(f"   Live: {len(live_df)} days")
            print(f"   Combined range: {merged_df.index[0].strftime('%Y-%m-%d')} to {merged_df.index[-1].strftime('%Y-%m-%d')}")
            
            return merged_df
            
        except Exception as e:
            print(f"❌ Data merge failed: {e}")
            return None
    
    def get_complete_dataset(self, historical_days=100):
        """
        Get complete dataset for strategy analysis
        """
        print("📊 BUILDING COMPLETE DATASET")
        print("=" * 40)
        
        # Fetch historical data
        historical_df = self.fetch_historical_data_coingecko(days=historical_days)
        
        # Add delay to be nice to APIs
        time.sleep(1)
        
        # Fetch live data
        live_df = self.fetch_live_data_binance()
        
        # Merge data
        complete_df = self.merge_historical_and_live(historical_df, live_df)
        
        if complete_df is not None:
            print(f"\n🎯 FINAL DATASET READY:")
            print(f"   Total days: {len(complete_df)}")
            print(f"   Suitable for 60-day windows: {'✅' if len(complete_df) >= 60 else '❌'}")
            print(f"   Current price: ${complete_df['close'].iloc[-1]:.2f}")
            
            # Test strategy compatibility
            self.test_strategy_compatibility(complete_df)
            
        return complete_df
    
    def test_strategy_compatibility(self, df):
        """
        Test if data is suitable for your strategy
        """
        print(f"\n🧠 STRATEGY COMPATIBILITY TEST:")
        print("=" * 30)
        
        windows = {'short': 10, 'medium': 30, 'long': 60}
        
        for name, window in windows.items():
            if len(df) >= window:
                print(f"✅ {name.title()} window ({window} days): Supported")
            else:
                print(f"❌ {name.title()} window ({window} days): Need {window - len(df)} more days")
        
        if len(df) >= 60:
            print("\n🎉 Perfect! Ready for full strategy testing")
            
            # Quick regime analysis
            self.quick_regime_analysis(df)
            
        else:
            print(f"\n⚠️  Limited data - consider shorter windows")
    
    def quick_regime_analysis(self, df):
        """
        Quick analysis using your strategy logic
        """
        try:
            print(f"\n🔍 QUICK REGIME ANALYSIS:")
            print("=" * 25)
            
            current_price = df['close'].iloc[-1]
            
            # Your strategy windows
            short_ma = df['close'].tail(10).mean()
            medium_ma = df['close'].tail(30).mean()
            long_ma = df['close'].tail(60).mean()
            
            # Trend analysis
            if current_price > long_ma and short_ma > medium_ma:
                trend = "Bullish"
            elif current_price < long_ma and short_ma < medium_ma:
                trend = "Bearish"
            else:
                trend = "Sideways"
            
            # Volatility
            returns = df['close'].pct_change().tail(14).dropna()
            volatility = returns.std()
            
            if volatility > 0.05:
                vol_regime = "High"
            elif volatility > 0.03:
                vol_regime = "Normal"
            else:
                vol_regime = "Low"
            
            # Position in range
            recent_high = df['close'].tail(30).max()
            recent_low = df['close'].tail(30).min()
            position_in_range = (current_price - recent_low) / (recent_high - recent_low)
            
            print(f"Current Price: ${current_price:.2f}")
            print(f"Short MA (10d): ${short_ma:.2f}")
            print(f"Medium MA (30d): ${medium_ma:.2f}")
            print(f"Long MA (60d): ${long_ma:.2f}")
            print(f"Trend: {trend}")
            print(f"Volatility: {vol_regime} ({volatility:.3f})")
            print(f"Position in 30d range: {position_in_range:.1%}")
            
            # Trading signal
            oversold_threshold = 0.3  # Bottom 30% of range
            overbought_threshold = 0.7  # Top 70% of range
            
            if position_in_range < oversold_threshold and trend != "Bearish":
                signal = "🟢 POTENTIAL BUY (oversold)"
            elif position_in_range > overbought_threshold:
                signal = "🔴 POTENTIAL SELL (overbought)" 
            else:
                signal = "🟡 HOLD (neutral)"
            
            print(f"Signal: {signal}")
            
            return {
                'current_price': current_price,
                'trend': trend,
                'volatility': vol_regime,
                'position_in_range': position_in_range,
                'signal': signal
            }
            
        except Exception as e:
            print(f"❌ Regime analysis failed: {e}")
            return None

def test_hybrid_pipeline():
    """
    Test the hybrid data pipeline
    """
    pipeline = HybridDataPipeline()
    
    # Get complete dataset
    df = pipeline.get_complete_dataset(historical_days=90)
    
    if df is not None:
        print(f"\n🚀 SUCCESS! Ready to integrate with your strategy")
        print(f"   Dataset size: {len(df)} days")
        print(f"   Ready for regime detection: {'✅' if len(df) >= 60 else '❌'}")
        
        # Save for later use
        df.to_csv('hybrid_avax_data.csv')
        print(f"   Saved to: hybrid_avax_data.csv")
        
        return df
    else:
        print(f"❌ Failed to create complete dataset")
        return None

if __name__ == "__main__":
    test_hybrid_pipeline()