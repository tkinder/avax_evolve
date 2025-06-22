# test_data_availability.py
import ccxt
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

print("📊 TESTING HISTORICAL DATA AVAILABILITY")
print("=" * 50)

# Initialize exchange
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_TESTNET_API_KEY'),
    'secret': os.getenv('BINANCE_TESTNET_SECRET'),
    'sandbox': True,
    'enableRateLimit': True,
})

def test_data_limits():
    """Test how much historical data we can get"""
    symbol = 'AVAX/USDT'
    timeframes = ['1d', '4h', '1h']
    limits_to_test = [50, 100, 200, 500, 1000]
    
    for timeframe in timeframes:
        print(f"\n📈 Testing {timeframe} timeframe:")
        
        for limit in limits_to_test:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    start_date = df['timestamp'].iloc[0].strftime('%Y-%m-%d')
                    end_date = df['timestamp'].iloc[-1].strftime('%Y-%m-%d')
                    days_span = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
                    
                    print(f"   ✅ Limit {limit:4d}: {len(ohlcv):3d} candles, {days_span:3d} days ({start_date} to {end_date})")
                else:
                    print(f"   ❌ Limit {limit:4d}: No data returned")
                    
            except Exception as e:
                print(f"   ❌ Limit {limit:4d}: Error - {e}")
                break  # Stop testing higher limits if this one fails

def get_optimal_data_for_strategy():
    """Get the best available data for your strategy"""
    symbol = 'AVAX/USDT'
    
    print(f"\n🎯 OPTIMIZING DATA FOR YOUR STRATEGY")
    print("=" * 40)
    
    # Try to get as much daily data as possible
    max_data = None
    best_limit = 0
    
    for limit in [1000, 500, 200, 100, 50]:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=limit)
            if ohlcv and len(ohlcv) > best_limit:
                max_data = ohlcv
                best_limit = len(ohlcv)
                print(f"✅ Successfully got {len(ohlcv)} days with limit={limit}")
                break
        except:
            continue
    
    if max_data:
        df = pd.DataFrame(max_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"\n📊 BEST AVAILABLE DATA:")
        print(f"   Total days: {len(df)}")
        print(f"   Date range: {df['timestamp'].iloc[0].strftime('%Y-%m-%d')} to {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # Check what strategy windows we can support
        print(f"\n🧠 STRATEGY COMPATIBILITY:")
        
        your_windows = {'short': 10, 'medium': 30, 'long': 60}
        
        for name, window in your_windows.items():
            if len(df) >= window:
                print(f"   ✅ {name.title()} window ({window} days): Supported")
            else:
                print(f"   ❌ {name.title()} window ({window} days): Need {window - len(df)} more days")
        
        # Suggest alternatives if needed
        if len(df) < 60:
            print(f"\n💡 ALTERNATIVE CONFIGURATIONS:")
            
            if len(df) >= 30:
                print(f"   Option A: Use windows 5/15/30 (conservative)")
                print(f"   Option B: Use windows 7/20/{len(df)} (adaptive)")
            elif len(df) >= 20:
                print(f"   Option A: Use windows 3/10/20 (short-term)")
                print(f"   Option B: Use windows 5/10/{len(df)} (minimal)")
            else:
                print(f"   ⚠️  Very limited - consider supplementing with external data")
        
        # Test regime detection with available data
        print(f"\n🔍 TESTING REGIME DETECTION:")
        
        try:
            # Simple regime analysis with available data
            current_price = df['close'].iloc[-1]
            
            # Use what we have
            available_days = min(len(df), 30)
            recent_data = df.tail(available_days)
            
            ma_short = recent_data['close'].tail(min(10, len(recent_data))).mean()
            ma_long = recent_data['close'].mean()
            
            trend = "Bullish" if current_price > ma_long else "Bearish"
            momentum = "Up" if current_price > ma_short else "Down"
            
            # Volatility
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            print(f"   Current Price: ${current_price:.2f}")
            print(f"   Short MA: ${ma_short:.2f}")
            print(f"   Long MA: ${ma_long:.2f}")
            print(f"   Trend: {trend}")
            print(f"   Momentum: {momentum}")
            print(f"   Volatility: {volatility:.4f}")
            
            # Check if this would be a buy signal with limited data
            price_vs_ma = (current_price - ma_long) / ma_long
            if price_vs_ma < -0.05:  # 5% below long MA
                signal = "POTENTIAL BUY (oversold)"
            elif price_vs_ma > 0.05:
                signal = "POTENTIAL SELL (overbought)"
            else:
                signal = "HOLD (neutral)"
            
            print(f"   Signal: {signal}")
            
        except Exception as e:
            print(f"   ❌ Regime detection test failed: {e}")
        
        return df
    else:
        print("❌ Could not retrieve sufficient data")
        return None

if __name__ == "__main__":
    # Test data availability
    test_data_limits()
    
    # Get optimal data for strategy
    df = get_optimal_data_for_strategy()
    
    print(f"\n🎯 RECOMMENDATIONS:")
    
    if df is not None and len(df) >= 60:
        print("✅ Perfect! You have enough data for full strategy testing")
    elif df is not None and len(df) >= 30:
        print("⚠️  Good! You can test with modified windows")
        print("   Consider using 5/15/30 day windows instead of 10/30/60")
    elif df is not None and len(df) >= 20:
        print("⚠️  Limited! You'll need to use very short windows")
        print("   Consider using 3/7/15 day windows for testing")
    else:
        print("❌ Insufficient data for reliable strategy testing")
        print("   Options:")
        print("   1. Use external data source (CoinGecko, Yahoo Finance)")
        print("   2. Wait for more testnet data to accumulate")
        print("   3. Test with Binance.US production data (read-only)")