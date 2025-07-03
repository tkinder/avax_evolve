# bitcoin_data_pipeline.py
import ccxt
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

load_dotenv()

class BitcoinDataPipeline:
    """
    Bitcoin-specific data pipeline combining external historical data with live Binance testnet data
    Adapted from HybridDataPipeline for Bitcoin optimization
    """
    
    def __init__(self):
        # Initialize Binance testnet for live data
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_TESTNET_API_KEY'),
            'secret': os.getenv('BINANCE_TESTNET_SECRET'),
            'sandbox': True,
            'enableRateLimit': True,
        })
        
        print("₿ Bitcoin Data Pipeline Initialized")
        print("📊 External data: CoinGecko (historical Bitcoin)")
        print("🎯 Live data: Binance Testnet (current BTC)")
        
    def fetch_historical_data_coingecko(self, symbol='bitcoin', days=100):
        """
        Fetch historical Bitcoin data from CoinGecko (free API)
        """
        try:
            print(f"📥 Fetching {days} days of Bitcoin historical data from CoinGecko...")
            
            # CoinGecko API endpoint for Bitcoin
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
            volumes = data['total_volumes']
            
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            
            # Merge price and volume data
            df = df.merge(volume_df, on='timestamp', how='left')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Create OHLC data (CoinGecko only gives daily close prices)
            # For backtesting, we'll approximate OHLC from close prices
            df['open'] = df['close'].shift(1)
            df['high'] = df[['open', 'close']].max(axis=1) * 1.015  # Bitcoin typically has smaller daily ranges than AVAX
            df['low'] = df[['open', 'close']].min(axis=1) * 0.985   # More conservative range for Bitcoin
            
            # Fill any missing volume with reasonable estimates
            df['volume'] = df['volume'].fillna(df['volume'].median())
            
            # Remove first row (no open price)
            df = df.dropna()
            
            # Reorder columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            print(f"✅ Got {len(df)} days of Bitcoin data from CoinGecko")
            print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
            print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ CoinGecko Bitcoin data fetch failed: {e}")
            return None
    
    def fetch_live_data_binance(self, symbol='BTC/USDT', limit=20):
        """
        Fetch recent Bitcoin data from Binance testnet
        """
        try:
            print(f"🎯 Fetching live Bitcoin data from Binance testnet...")
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1d', limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ Got {len(df)} days of Bitcoin from Binance testnet")
            print(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")
            # Safe 24h change calculation
            if len(df) >= 2:
                change_pct = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100
                print(f"   24h change: {change_pct:+.2f}%")
            else:
                print(f"   24h change: N/A (insufficient data)")
            
            return df
            
        except Exception as e:
            print(f"❌ Binance Bitcoin live data fetch failed: {e}")
            return None
    
    def merge_historical_and_live(self, historical_df, live_df):
        """
        Merge historical and live Bitcoin data, avoiding duplicates
        """
        try:
            print("🔄 Merging Bitcoin historical and live data...")
            
            # Handle case where live data failed to fetch
            if historical_df is None:
                print("❌ Cannot merge - missing historical Bitcoin data")
                return None
                
            if live_df is None:
                print("⚠️ No live data available - using historical data only")
                print(f"✅ Using historical Bitcoin data: {len(historical_df)} days")
                print(f"   Range: {historical_df.index[0].strftime('%Y-%m-%d')} to {historical_df.index[-1].strftime('%Y-%m-%d')}")
                return historical_df
            
            # Find overlap and remove duplicates
            live_start = live_df.index[0]
            
            # Keep historical data up to live data start
            historical_clean = historical_df[historical_df.index < live_start]
            
            # Combine
            merged_df = pd.concat([historical_clean, live_df])
            merged_df = merged_df.sort_index()
            
            print(f"✅ Merged Bitcoin data: {len(merged_df)} total days")
            print(f"   Historical: {len(historical_clean)} days")
            print(f"   Live: {len(live_df)} days")
            print(f"   Combined range: {merged_df.index[0].strftime('%Y-%m-%d')} to {merged_df.index[-1].strftime('%Y-%m-%d')}")
            
            return merged_df
            
        except Exception as e:
            print(f"❌ Bitcoin data merge failed: {e}")
            # Fallback to historical data only
            if historical_df is not None:
                print(f"🔄 Falling back to historical data only")
                return historical_df
            return None
    
    def get_complete_dataset(self, historical_days=100):
        """
        Get complete Bitcoin dataset for strategy analysis
        """
        print("₿ BUILDING COMPLETE BITCOIN DATASET")
        print("=" * 45)
        
        # Fetch historical Bitcoin data
        historical_df = self.fetch_historical_data_coingecko(days=historical_days)
        
        # Add delay to be nice to APIs
        time.sleep(1)
        
        # Fetch live Bitcoin data
        live_df = self.fetch_live_data_binance()
        
        # Merge data
        complete_df = self.merge_historical_and_live(historical_df, live_df)
        
        if complete_df is not None:
            print(f"\n🎯 FINAL BITCOIN DATASET READY:")
            print(f"   Total days: {len(complete_df)}")
            print(f"   Suitable for 60-day windows: {'✅' if len(complete_df) >= 60 else '❌'}")
            print(f"   Current price: ${complete_df['close'].iloc[-1]:,.2f}")
            print(f"   Price volatility: {complete_df['close'].pct_change().std() * 100:.2f}%")
            
            # Test strategy compatibility
            self.test_strategy_compatibility(complete_df)
            
        return complete_df
    
    def test_strategy_compatibility(self, df):
        """
        Test if Bitcoin data is suitable for strategy optimization
        """
        print(f"\n🧠 BITCOIN STRATEGY COMPATIBILITY TEST:")
        print("=" * 40)
        
        windows = {'short': 10, 'medium': 30, 'long': 60}
        
        for name, window in windows.items():
            if len(df) >= window:
                print(f"✅ {name.title()} window ({window} days): Supported")
            else:
                print(f"❌ {name.title()} window ({window} days): Need {window - len(df)} more days")
        
        # Check data quality for Bitcoin
        print(f"\n📊 BITCOIN DATA QUALITY:")
        price_changes = df['close'].pct_change().dropna()
        
        print(f"   Daily volatility: {price_changes.std() * 100:.2f}%")
        print(f"   Max daily gain: {price_changes.max() * 100:+.2f}%")
        print(f"   Max daily loss: {price_changes.min() * 100:+.2f}%")
        print(f"   Trend consistency: {'High' if abs(price_changes.mean()) > 0.001 else 'Low'}")
        
        if len(df) >= 60:
            print("\n🎉 Perfect! Ready for Bitcoin parameter optimization")
            
            # Bitcoin-specific regime analysis
            self.bitcoin_regime_analysis(df)
            
        else:
            print(f"\n⚠️  Limited data - consider shorter windows for Bitcoin")
    
    def bitcoin_regime_analysis(self, df):
        """
        Bitcoin-specific market regime analysis
        """
        try:
            print(f"\n₿ BITCOIN REGIME ANALYSIS:")
            print("=" * 30)
            
            current_price = df['close'].iloc[-1]
            
            # Bitcoin-specific moving averages
            short_ma = df['close'].tail(10).mean()
            medium_ma = df['close'].tail(30).mean()
            long_ma = df['close'].tail(60).mean()
            
            # Bitcoin trend analysis (more sensitive to volatility)
            if current_price > long_ma * 1.02 and short_ma > medium_ma:  # 2% buffer for Bitcoin
                trend = "Bullish"
            elif current_price < long_ma * 0.98 and short_ma < medium_ma:
                trend = "Bearish"
            else:
                trend = "Sideways"
            
            # Bitcoin volatility analysis
            returns = df['close'].pct_change().tail(14).dropna()
            volatility = returns.std()
            
            # Bitcoin-specific volatility thresholds
            if volatility > 0.04:  # 4%+ daily volatility
                vol_regime = "High"
            elif volatility > 0.025:  # 2.5%+ daily volatility
                vol_regime = "Normal"
            else:
                vol_regime = "Low"
            
            # Position in recent range
            recent_high = df['close'].tail(30).max()
            recent_low = df['close'].tail(30).min()
            position_in_range = (current_price - recent_low) / (recent_high - recent_low)
            
            # Bitcoin support/resistance levels
            support_level = df['close'].tail(60).quantile(0.25)
            resistance_level = df['close'].tail(60).quantile(0.75)
            
            print(f"Current Price: ${current_price:,.2f}")
            print(f"Short MA (10d): ${short_ma:,.2f}")
            print(f"Medium MA (30d): ${medium_ma:,.2f}")
            print(f"Long MA (60d): ${long_ma:,.2f}")
            print(f"Trend: {trend}")
            print(f"Volatility: {vol_regime} ({volatility:.3f})")
            print(f"Position in 30d range: {position_in_range:.1%}")
            print(f"Support level: ${support_level:,.2f}")
            print(f"Resistance level: ${resistance_level:,.2f}")
            
            # Bitcoin trading signal (more conservative than AVAX)
            oversold_threshold = 0.25  # Bottom 25% of range (more conservative for Bitcoin)
            overbought_threshold = 0.75  # Top 75% of range
            
            if position_in_range < oversold_threshold and trend != "Bearish":
                signal = "🟢 POTENTIAL BUY (Bitcoin oversold)"
            elif position_in_range > overbought_threshold:
                signal = "🔴 POTENTIAL SELL (Bitcoin overbought)" 
            else:
                signal = "🟡 HOLD (Bitcoin neutral)"
            
            print(f"Signal: {signal}")
            
            # Calculate preliminary parameter suggestions for optimization
            price_range = recent_high - recent_low
            range_pct = price_range / current_price
            
            # Suggest initial parameter ranges based on Bitcoin volatility
            suggested_buy_threshold = max(0.15, min(0.35, 0.3 - (volatility * 2)))
            suggested_sell_threshold = min(0.85, max(0.65, 0.7 + (volatility * 2)))
            
            print(f"\n🎯 PRELIMINARY PARAMETER SUGGESTIONS:")
            print(f"   Suggested buy threshold: {suggested_buy_threshold:.2f}")
            print(f"   Suggested sell threshold: {suggested_sell_threshold:.2f}")
            print(f"   30-day price range: {range_pct:.1%}")
            print(f"   💡 DEAP will optimize these parameters scientifically!")
            
            return {
                'current_price': current_price,
                'trend': trend,
                'volatility': vol_regime,
                'position_in_range': position_in_range,
                'signal': signal,
                'suggested_buy_threshold': suggested_buy_threshold,
                'suggested_sell_threshold': suggested_sell_threshold,
                'support': support_level,
                'resistance': resistance_level
            }
            
        except Exception as e:
            print(f"❌ Bitcoin regime analysis failed: {e}")
            return None

def test_bitcoin_pipeline():
    """
    Test the Bitcoin data pipeline
    """
    pipeline = BitcoinDataPipeline()
    
    # Get complete Bitcoin dataset
    df = pipeline.get_complete_dataset(historical_days=90)
    
    if df is not None:
        print(f"\n🚀 SUCCESS! Bitcoin data pipeline ready for DEAP optimization!")
        print(f"   Dataset size: {len(df)} days")
        print(f"   Ready for regime detection: {'✅' if len(df) >= 60 else '❌'}")
        print(f"   Price range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
        
        # Save for DEAP optimization
        df.to_csv('bitcoin_data.csv')
        print(f"   Saved to: bitcoin_data.csv")
        
        # Calculate some Bitcoin-specific metrics for DEAP
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        
        print(f"\n₿ BITCOIN CHARACTERISTICS FOR OPTIMIZATION:")
        print(f"   Average daily volatility: {volatility:.3f}")
        print(f"   Max daily gain: {returns.max():.3f}")
        print(f"   Max daily loss: {returns.min():.3f}")
        print(f"   Trend strength: {abs(returns.mean()):.5f}")
        
        print(f"\n🧬 Ready for DEAP parameter optimization!")
        print(f"   DEAP will test 1,250+ parameter combinations")
        print(f"   Target: Find optimal Bitcoin buy/sell thresholds")
        print(f"   Expected runtime: 30-45 minutes")
        
        return df
    else:
        print(f"❌ Failed to create Bitcoin dataset")
        return None

if __name__ == "__main__":
    test_bitcoin_pipeline()