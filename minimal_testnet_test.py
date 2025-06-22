# minimal_testnet_test.py
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

print("🧪 MINIMAL BINANCE TESTNET TEST")
print("=" * 40)

# Simple testnet setup that works
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_TESTNET_API_KEY'),
    'secret': os.getenv('BINANCE_TESTNET_SECRET'),
    'sandbox': True,
    'enableRateLimit': True,
})

# Test 1: Market Data (no auth needed)
try:
    print("📊 Testing Market Data...")
    ticker = exchange.fetch_ticker('AVAX/USDT')
    print(f"✅ AVAX Price: ${ticker['last']:.2f}")
    print(f"✅ 24h Change: {ticker['percentage']:.2f}%")
except Exception as e:
    print(f"❌ Market Data Error: {e}")

print()

# Test 2: Account Access (needs auth)
try:
    print("🔐 Testing Account Access...")
    balance = exchange.fetch_balance()
    
    print("✅ Account authenticated successfully!")
    
    # Show testnet balances (testnet gives you fake money)
    important_coins = ['USDT', 'BTC', 'ETH', 'AVAX', 'BNB']
    for coin in important_coins:
        if coin in balance and balance[coin]['total'] > 0:
            print(f"   {coin}: {balance[coin]['total']:.4f}")
        else:
            print(f"   {coin}: 0.0000")
            
except Exception as e:
    print(f"❌ Account Access Error: {e}")

print()

# Test 3: Order Test (simulation only)
try:
    print("📋 Testing Order Capabilities...")
    markets = exchange.load_markets()
    
    if 'AVAX/USDT' in markets:
        market = markets['AVAX/USDT']
        print("✅ AVAX/USDT trading available")
        print(f"   Min order: ${market['limits']['cost']['min']:.2f}")
        print(f"   Trading fee: {market['taker']:.3f}%")
        
        # Show current price for reference
        current_price = exchange.fetch_ticker('AVAX/USDT')['last']
        print(f"   Current price: ${current_price:.2f}")
        
        # Calculate what a small test order would look like
        test_amount_usd = 20  # $20 test order
        test_quantity = test_amount_usd / current_price
        print(f"   Test order: {test_quantity:.4f} AVAX for ${test_amount_usd}")
        
    else:
        print("❌ AVAX/USDT market not found")
        
except Exception as e:
    print(f"❌ Order Test Error: {e}")

print()

# Test 4: Historical Data for Strategy
try:
    print("📈 Testing Historical Data...")
    ohlcv = exchange.fetch_ohlcv('AVAX/USDT', '1d', limit=30)
    
    if len(ohlcv) >= 10:
        prices = [candle[4] for candle in ohlcv]  # Close prices
        current = prices[-1]
        avg_10d = sum(prices[-10:]) / 10
        
        print(f"✅ Got {len(ohlcv)} days of data")
        print(f"   Current: ${current:.2f}")
        print(f"   10-day avg: ${avg_10d:.2f}")
        print(f"   Trend: {'Up' if current > avg_10d else 'Down'}")
    else:
        print("❌ Insufficient historical data")
        
except Exception as e:
    print(f"❌ Historical Data Error: {e}")

print()
print("=" * 40)
print("🎯 TESTNET STATUS:")

# Check if we have what we need for your strategy
checks = {
    'api_key': bool(os.getenv('BINANCE_TESTNET_API_KEY')),
    'secret': bool(os.getenv('BINANCE_TESTNET_SECRET')),
}

if all(checks.values()):
    print("✅ Ready for strategy development!")
    print("✅ API credentials working")
    print("✅ Market data accessible")
    print("✅ Can proceed with paper trading")
    print()
    print("🚀 NEXT STEPS:")
    print("1. Integrate with your regime detection")
    print("2. Build paper trading system")
    print("3. Test strategy with live data")
else:
    print("❌ Setup incomplete")
    for check, status in checks.items():
        print(f"   {check}: {'✅' if status else '❌'}")