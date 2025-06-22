# binance_us_test.py
import ccxt
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# binance_test.py - Updated for correct exchange type
import ccxt
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

class BinanceAPITest:
    def __init__(self, use_testnet=True):
        """
        Initialize with either Binance.com testnet or Binance.US production
        
        Args:
            use_testnet (bool): True for Binance.com testnet, False for Binance.US production
        """
        self.use_testnet = use_testnet
        
        if use_testnet:
            # Binance.com testnet (safe fake money)
            self.exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_TESTNET_API_KEY'),
                'secret': os.getenv('BINANCE_TESTNET_SECRET'),
                'enableRateLimit': True,
                'sandbox': True,
                'urls': {
                    'api': {
                        'public': 'https://testnet.binance.vision/api',
                        'private': 'https://testnet.binance.vision/api',
                    }
                }
            })
            print("🧪 Binance.com TESTNET Initialized (SAFE - Fake Money)")
            print("📍 Using: testnet.binance.vision")
        else:
            # Binance.US production (real money)
            self.exchange = ccxt.binanceus({
                'apiKey': os.getenv('BINANCE_US_API_KEY'),
                'secret': os.getenv('BINANCE_US_SECRET'),
                'enableRateLimit': True,
                'sandbox': False,  # No testnet available for Binance.US
            })
            print("💰 Binance.US PRODUCTION Initialized (REAL MONEY)")
            print("📍 Using: binance.us")
            
        print(f"📅 Test Time: {datetime.now()}")
        print("=" * 50)
    
    def test_connection(self):
        """Test basic API connectivity"""
        try:
            # Test if we can connect (this doesn't require API keys)
            status = self.exchange.fetch_status()
            print("✅ Connection Test:")
            print(f"   Exchange Status: {status['status']}")
            print(f"   Exchange Updated: {status['updated']}")
            return True
        except Exception as e:
            print(f"❌ Connection Test Failed: {e}")
            return False
    
    def test_account_access(self):
        """Test if API keys work for account access"""
        try:
            # This requires valid API keys
            balance = self.exchange.fetch_balance()
            print("✅ Account Access Test:")
            print(f"   Total USD Balance: ${balance['USD']['total']:.2f}")
            print(f"   Free USD Balance: ${balance['USD']['free']:.2f}")
            
            # Check if we have any AVAX
            if 'AVAX' in balance and balance['AVAX']['total'] > 0:
                print(f"   AVAX Balance: {balance['AVAX']['total']:.4f} AVAX")
            else:
                print("   AVAX Balance: 0.0000 AVAX")
            
            return True
        except Exception as e:
            print(f"❌ Account Access Test Failed: {e}")
            return False
    
    def test_market_data(self):
        """Test fetching market data (no API keys required)"""
        try:
            # Get current price (use appropriate symbol for exchange)
            symbol = 'AVAX/USDT' if self.use_testnet else 'AVAX/USD'
            ticker = self.exchange.fetch_ticker(symbol)
            print("✅ Market Data Test:")
            print(f"   Current AVAX Price: ${ticker['last']:.2f}")
            print(f"   24h Change: {ticker['percentage']:.2f}%")
            print(f"   24h Volume: {ticker['baseVolume']:.0f} AVAX")
            print(f"   Trading Pair: {symbol}")
            
            # Get some historical data for regime detection
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1d', limit=10)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            print(f"   Historical Data: {len(df)} days retrieved")
            print(f"   Price Range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            return True, df
        except Exception as e:
            print(f"❌ Market Data Test Failed: {e}")
            return False, None
    
    def test_trading_permissions(self):
        """Test trading permissions (without actually placing orders)"""
        try:
            # Check what trading pairs are available
            markets = self.exchange.load_markets()
            symbol = 'AVAX/USDT' if self.use_testnet else 'AVAX/USD'
            
            if symbol in markets:
                market_info = markets[symbol]
                print("✅ Trading Permissions Test:")
                print(f"   {symbol} Market: Available")
                print(f"   Min Order Size: ${market_info['limits']['cost']['min']:.2f}")
                print(f"   Trading Fee: {market_info['taker']:.3f}%")
                return True
            else:
                print(f"❌ {symbol} market not available")
                return False
                
        except Exception as e:
            print(f"❌ Trading Permissions Test Failed: {e}")
            return False
    
    def test_regime_detection_integration(self, df):
        """Test if we can run regime detection on live data"""
        try:
            # Simple regime detection test (without importing your full system)
            if df is None or len(df) < 10:
                print("❌ Insufficient data for regime detection")
                return False
            
            # Basic trend calculation
            current_price = df['close'].iloc[-1]
            ma_10 = df['close'].tail(10).mean()
            price_change = (current_price - df['close'].iloc[-10]) / df['close'].iloc[-10]
            
            print("✅ Regime Detection Test:")
            print(f"   Current Price: ${current_price:.2f}")
            print(f"   10-day Average: ${ma_10:.2f}")
            print(f"   10-day Change: {price_change:.2f}%")
            
            # Simple trend classification
            if price_change > 0.1:
                trend = "Bullish"
            elif price_change < -0.1:
                trend = "Bearish"
            else:
                trend = "Sideways"
            
            print(f"   Simple Trend: {trend}")
            return True
            
        except Exception as e:
            print(f"❌ Regime Detection Test Failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        exchange_type = "TESTNET" if self.use_testnet else "PRODUCTION"
        print(f"🧪 BINANCE {exchange_type} API TEST SUITE")
        print("=" * 50)
        
        tests_passed = 0
        total_tests = 5
        
        # Test 1: Basic Connection
        if self.test_connection():
            tests_passed += 1
        
        print()
        
        # Test 2: Account Access
        if self.test_account_access():
            tests_passed += 1
        
        print()
        
        # Test 3: Market Data
        success, df = self.test_market_data()
        if success:
            tests_passed += 1
        
        print()
        
        # Test 4: Trading Permissions
        if self.test_trading_permissions():
            tests_passed += 1
        
        print()
        
        # Test 5: Regime Detection Integration
        if self.test_regime_detection_integration(df):
            tests_passed += 1
        
        print()
        print("=" * 50)
        print(f"🎯 TEST RESULTS: {tests_passed}/{total_tests} PASSED")
        
        if tests_passed == total_tests:
            if self.use_testnet:
                print("🎉 ALL TESTS PASSED! Testnet ready for development.")
            else:
                print("🎉 ALL TESTS PASSED! Ready for REAL MONEY trading.")
                print("⚠️  BE CAREFUL: This is your real Binance.US account!")
        elif tests_passed >= 3:
            print("⚠️  Most tests passed. Check failed tests above.")
        else:
            print("❌ Multiple test failures. Check API keys and permissions.")
        
        return tests_passed == total_tests

# Paper Trading Simulator for safe testing
class BinanceUSPaperTrader:
    def __init__(self):
        self.exchange = ccxt.binanceus()  # No API keys needed for market data
        self.paper_balance = 10000.0  # Start with $10k fake money
        self.paper_positions = {}
        self.trade_log = []
        print("📝 Paper Trading Mode Initialized")
        print(f"💰 Starting Balance: ${self.paper_balance:.2f}")
    
    def get_current_price(self, symbol='AVAX/USD'):
        """Get real current price from Binance.US"""
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']
    
    def simulate_buy(self, amount_usd, symbol='AVAX/USD'):
        """Simulate buying AVAX with real prices"""
        try:
            current_price = self.get_current_price(symbol)
            
            if amount_usd > self.paper_balance:
                return {'success': False, 'error': 'Insufficient balance'}
            
            # Simulate realistic execution
            executed_price = current_price * 1.001  # 0.1% slippage
            fee = amount_usd * 0.001  # 0.1% fee
            quantity = amount_usd / executed_price
            total_cost = amount_usd + fee
            
            # Update paper portfolio
            self.paper_balance -= total_cost
            self.paper_positions[symbol] = self.paper_positions.get(symbol, 0) + quantity
            
            trade = {
                'type': 'BUY',
                'timestamp': datetime.now(),
                'symbol': symbol,
                'quantity': quantity,
                'price': executed_price,
                'amount': amount_usd,
                'fee': fee,
                'balance': self.paper_balance
            }
            self.trade_log.append(trade)
            
            print(f"📈 PAPER BUY: {quantity:.4f} AVAX at ${executed_price:.2f}")
            print(f"💰 Remaining Balance: ${self.paper_balance:.2f}")
            
            return {'success': True, 'trade': trade}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def simulate_sell(self, quantity, symbol='AVAX/USD'):
        """Simulate selling AVAX with real prices"""
        try:
            current_position = self.paper_positions.get(symbol, 0)
            
            if quantity > current_position:
                return {'success': False, 'error': 'Insufficient AVAX position'}
            
            current_price = self.get_current_price(symbol)
            executed_price = current_price * 0.999  # 0.1% slippage
            gross_proceeds = quantity * executed_price
            fee = gross_proceeds * 0.001  # 0.1% fee
            net_proceeds = gross_proceeds - fee
            
            # Update paper portfolio
            self.paper_balance += net_proceeds
            self.paper_positions[symbol] -= quantity
            
            trade = {
                'type': 'SELL',
                'timestamp': datetime.now(),
                'symbol': symbol,
                'quantity': quantity,
                'price': executed_price,
                'amount': gross_proceeds,
                'fee': fee,
                'balance': self.paper_balance
            }
            self.trade_log.append(trade)
            
            print(f"📉 PAPER SELL: {quantity:.4f} AVAX at ${executed_price:.2f}")
            print(f"💰 New Balance: ${self.paper_balance:.2f}")
            
            return {'success': True, 'trade': trade}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_portfolio_status(self):
        """Get current portfolio status"""
        try:
            total_value = self.paper_balance
            
            for symbol, quantity in self.paper_positions.items():
                if quantity > 0:
                    current_price = self.get_current_price(symbol)
                    position_value = quantity * current_price
                    total_value += position_value
                    
                    print(f"📊 {symbol}: {quantity:.4f} @ ${current_price:.2f} = ${position_value:.2f}")
            
            print(f"💵 Cash: ${self.paper_balance:.2f}")
            print(f"💎 Total Portfolio: ${total_value:.2f}")
            print(f"📈 Total Return: {((total_value - 10000) / 10000 * 100):.2f}%")
            
            return total_value
            
        except Exception as e:
            print(f"Error getting portfolio status: {e}")
            return self.paper_balance

if __name__ == "__main__":
    print("🤔 Which API keys do you have?")
    print("1. Binance.com TESTNET keys (from testnet.binance.vision) - SAFE")
    print("2. Binance.US PRODUCTION keys (from binance.us) - REAL MONEY")
    
    choice = input("\nEnter 1 for testnet or 2 for production: ").strip()
    
    if choice == "1":
        print("\n🧪 Using Binance.com TESTNET - Safe to test!")
        print("Make sure your .env file has:")
        print("BINANCE_TESTNET_API_KEY=your_testnet_key")
        print("BINANCE_TESTNET_SECRET=your_testnet_secret")
        
        tester = BinanceAPITest(use_testnet=True)
        success = tester.run_all_tests()
        
    elif choice == "2":
        print("\n💰 Using Binance.US PRODUCTION - REAL MONEY!")
        print("⚠️  WARNING: This will access your real account!")
        print("Make sure your .env file has:")
        print("BINANCE_US_API_KEY=your_real_key")
        print("BINANCE_US_SECRET=your_real_secret")
        
        confirm = input("\nType 'YES' to proceed with real money testing: ").strip()
        if confirm == "YES":
            tester = BinanceAPITest(use_testnet=False)
            success = tester.run_all_tests()
        else:
            print("❌ Cancelled. Create testnet keys first for safe testing.")
            success = False
    else:
        print("❌ Invalid choice. Please run again and select 1 or 2.")
        success = False
    
    if success:
        print("\n" + "=" * 50)
        print("🚀 API Tests Passed - Ready for Next Steps!")
        print("=" * 50)