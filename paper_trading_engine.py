"""
AVAX Paper Trading Engine
========================
Executes your proven strategy on Binance testnet with real market data
Integrates with your existing monitoring and regime detection system
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd

# Import your existing components from src/core/
import sys
import os
sys.path.append('src')
sys.path.append('src/core')

try:
    from core.regime_detector import EnhancedRegimeDetector, RegimeBasedTradingRules
    from core.adaptive_backtester import AdaptivePriceLevels
    from hybrid_data_pipeline import HybridDataPipeline
    print("✅ Successfully imported strategy modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Available files in src/core/:")
    if os.path.exists('src/core'):
        print(os.listdir('src/core'))
    else:
        print("src/core directory not found")
    sys.exit(1)

class PaperTradingEngine:
    """
    Paper Trading Engine that executes your AVAX strategy on Binance testnet
    Integrates with your proven regime detection and adaptive price levels
    """
    
    def __init__(self, config_file='paper_trading_config.json'):
        """Initialize the paper trading engine"""
        self.config = self.load_config(config_file)
        self.setup_logging()
        
        # Initialize Binance testnet client
        self.client = Client(
            api_key=self.config['binance']['testnet_api_key'],
            api_secret=self.config['binance']['testnet_api_secret'],
            testnet=True
        )
        
        # Initialize your proven strategy components
        self.data_pipeline = HybridDataPipeline()
        self.regime_detector = EnhancedRegimeDetector()
        
        # Initialize AdaptivePriceLevels with your proven parameters
        try:
            from core.adaptive_backtester import AdaptiveParams
            self.adaptive_params = AdaptiveParams(
                buy_threshold_pct=0.162,      # 16.2% of range (your optimized value)
                sell_threshold_pct=0.691,     # 69.1% of range (your optimized value)
                regime_short_window=10,
                regime_medium_window=30,
                regime_long_window=60,
                max_position_pct=0.80,
                stop_loss_pct=0.082,          # 8.2% (your proven value)
                take_profit_pct=0.150,        # 15.0% (your proven value)
            )
            self.price_calculator = AdaptivePriceLevels(self.adaptive_params)
        except ImportError:
            # Fallback if AdaptiveParams not available
            print("⚠️ Using simplified price calculation")
            self.price_calculator = None
            self.adaptive_params = None
        
        # Trading state
        self.position = None
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'current_balance': self.config['trading']['initial_balance']
        }
        
        # Load existing state if available
        self.load_trading_state()
        
        self.logger.info("🚀 Paper Trading Engine initialized")
        self.logger.info(f"💰 Starting balance: ${self.performance_metrics['current_balance']:,.2f}")
    
    def load_config(self, config_file):
        """Load configuration or create default"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Create default config
        default_config = {
            "binance": {
                "testnet_api_key": "YOUR_TESTNET_API_KEY",
                "testnet_api_secret": "YOUR_TESTNET_API_SECRET"
            },
            "trading": {
                "symbol": "AVAXUSDT",
                "initial_balance": 10000.0,
                "max_position_pct": 0.95,
                "min_confidence": 0.60,
                "stop_loss_pct": 0.082,
                "take_profit_pct": 0.15
            },
            "risk_management": {
                "max_daily_loss": 0.05,
                "max_drawdown": 0.10,
                "min_order_size": 10.0
            },
            "alerts": {
                "email_on_trade": True,
                "email_on_signal": True
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        print(f"📝 Created default config: {config_file}")
        print("⚠️  Please update with your Binance testnet API keys!")
        
        return default_config
    
    def setup_logging(self):
        """Setup detailed logging for paper trading"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('paper_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_trading_state(self):
        """Load existing trading state from file"""
        state_file = 'trading_state.json'
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.position = state.get('position')
                    self.trade_history = state.get('trade_history', [])
                    self.performance_metrics.update(state.get('performance_metrics', {}))
                self.logger.info("📊 Loaded existing trading state")
            except Exception as e:
                self.logger.error(f"❌ Failed to load trading state: {e}")
    
    def save_trading_state(self):
        """Save current trading state to file"""
        state = {
            'position': self.position,
            'trade_history': self.trade_history,
            'performance_metrics': self.performance_metrics,
            'last_update': datetime.now().isoformat()
        }
        
        with open('trading_state.json', 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_account_balance(self):
        """Get current testnet account balance"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == 'USDT':
                    return float(balance['free'])
            return 0.0
        except Exception as e:
            self.logger.error(f"❌ Failed to get balance: {e}")
            return self.performance_metrics['current_balance']
    
    def get_current_price(self):
        """Get current AVAX price from Binance"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.config['trading']['symbol'])
            return float(ticker['price'])
        except Exception as e:
            self.logger.error(f"❌ Failed to get price: {e}")
            return None
    
    def analyze_market_conditions(self):
        """Run your proven strategy analysis"""
        try:
            # Get market data using your hybrid pipeline
            df = self.data_pipeline.get_complete_dataset(historical_days=90)
            if df is None or len(df) < 60:
                self.logger.error("❌ Insufficient data for analysis")
                return None
            
            current_price = self.get_current_price()
            if current_price is None:
                return None
            
            # Use your proven regime detection
            current_idx = len(df) - 1  # Last index for current analysis
            regime = self.regime_detector.detect_regime(df, current_idx)
            
            # Calculate adaptive price levels using the correct method
            try:
                buy_level, sell_level, support, resistance = self.price_calculator.get_price_levels(df, current_idx)
            except Exception as e:
                self.logger.warning(f"Price calculation failed: {e}, using fallback")
                # Fallback calculation
                current_price_df = df['close'].iloc[-1]
                recent_high = df['high'].rolling(30).max().iloc[-1]
                recent_low = df['low'].rolling(30).min().iloc[-1]
                price_range = recent_high - recent_low
                
                buy_level = recent_low + (price_range * 0.162)  # 16.2% of range
                sell_level = recent_low + (price_range * 0.691)  # 69.1% of range
                support = recent_low
                resistance = recent_high
            
            # Apply your trading rules
            basic_entry = RegimeBasedTradingRules.should_enter_long(regime, current_price, buy_level)
            confidence_ok = regime.confidence >= self.config['trading']['min_confidence']
            
            # Additional filters (from your laptop monitor)
            price_range = max(resistance - support, current_price * 0.01)
            price_position = max(0, min(1, (current_price - support) / price_range))
            
            momentum_filter = True
            if regime.momentum.value == "accel_down":
                momentum_filter = (price_position < 0.25 and regime.trend.value != "strong_bear")
            elif regime.momentum.value == "steady_down":
                momentum_filter = price_position < 0.4
            
            trend_filter = True
            if regime.trend.value in ["mild_bear", "strong_bear"]:
                trend_filter = price_position < 0.2
            
            should_enter = (basic_entry and confidence_ok and momentum_filter and 
                          trend_filter and not regime.should_avoid_trading())
            
            analysis = {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'regime': regime,
                'buy_level': buy_level,
                'sell_level': sell_level,
                'support': support,
                'resistance': resistance,
                'should_enter': should_enter,
                'confidence_ok': confidence_ok,
                'momentum_filter': momentum_filter,
                'trend_filter': trend_filter,
                'basic_entry': basic_entry,
                'price_position': price_position
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Market analysis failed: {e}")
            return None
    
    def calculate_position_size(self, current_price, regime):
        """Calculate position size based on your proven strategy"""
        try:
            # Get current balance
            balance = self.get_account_balance()
            
            # Base position size (95% of balance as per your config)
            base_size_usd = balance * self.config['trading']['max_position_pct']
            
            # Apply regime-based position multiplier (from your strategy)
            regime_multiplier = regime.get_position_multiplier()
            adjusted_size_usd = base_size_usd * regime_multiplier
            
            # Convert to AVAX quantity
            avax_quantity = adjusted_size_usd / current_price
            
            # Apply minimum order size
            min_size_usd = self.config['risk_management']['min_order_size']
            min_avax = min_size_usd / current_price
            
            final_quantity = max(avax_quantity, min_avax)
            
            self.logger.info(f"💰 Position sizing: ${adjusted_size_usd:.2f} → {final_quantity:.4f} AVAX")
            
            return final_quantity
            
        except Exception as e:
            self.logger.error(f"❌ Position sizing failed: {e}")
            return 0.0
    
    def place_buy_order(self, quantity, current_price):
        """Place buy order on Binance testnet"""
        try:
            self.logger.info(f"🟢 Placing BUY order: {quantity:.4f} AVAX at ~${current_price:.2f}")
            
            # Place market order
            order = self.client.order_market_buy(
                symbol=self.config['trading']['symbol'],
                quantity=f"{quantity:.4f}"
            )
            
            # Get actual fill price
            fill_price = float(order['fills'][0]['price']) if order['fills'] else current_price
            fill_quantity = float(order['executedQty'])
            
            self.logger.info(f"✅ BUY order filled: {fill_quantity:.4f} AVAX at ${fill_price:.2f}")
            
            return {
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': 'BUY',
                'quantity': fill_quantity,
                'price': fill_price,
                'timestamp': datetime.now(),
                'status': 'FILLED'
            }
            
        except BinanceAPIException as e:
            self.logger.error(f"❌ Binance API error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Buy order failed: {e}")
            return None
    
    def place_sell_order(self, quantity, current_price):
        """Place sell order on Binance testnet"""
        try:
            self.logger.info(f"🔴 Placing SELL order: {quantity:.4f} AVAX at ~${current_price:.2f}")
            
            # Place market order
            order = self.client.order_market_sell(
                symbol=self.config['trading']['symbol'],
                quantity=f"{quantity:.4f}"
            )
            
            # Get actual fill price
            fill_price = float(order['fills'][0]['price']) if order['fills'] else current_price
            fill_quantity = float(order['executedQty'])
            
            self.logger.info(f"✅ SELL order filled: {fill_quantity:.4f} AVAX at ${fill_price:.2f}")
            
            return {
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': 'SELL',
                'quantity': fill_quantity,
                'price': fill_price,
                'timestamp': datetime.now(),
                'status': 'FILLED'
            }
            
        except BinanceAPIException as e:
            self.logger.error(f"❌ Binance API error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Sell order failed: {e}")
            return None
    
    def execute_entry_signal(self, analysis):
        """Execute entry signal - buy AVAX"""
        if self.position is not None:
            self.logger.warning("⚠️ Already in position, skipping entry signal")
            return False
        
        current_price = analysis['current_price']
        regime = analysis['regime']
        
        # Calculate position size
        quantity = self.calculate_position_size(current_price, regime)
        if quantity <= 0:
            self.logger.error("❌ Invalid position size calculated")
            return False
        
        # Place buy order
        order = self.place_buy_order(quantity, current_price)
        if order is None:
            return False
        
        # Calculate stop loss and take profit levels
        entry_price = order['price']
        stop_loss = entry_price * (1 - self.config['trading']['stop_loss_pct'])
        take_profit = entry_price * (1 + self.config['trading']['take_profit_pct'])
        
        # Create position record
        self.position = {
            'entry_order': order,
            'entry_price': entry_price,
            'quantity': order['quantity'],
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_analysis': analysis,
            'days_held': 0
        }
        
        # Update performance metrics
        self.performance_metrics['total_trades'] += 1
        
        # Save state
        self.save_trading_state()
        
        self.logger.info(f"🎯 POSITION OPENED:")
        self.logger.info(f"   Entry: ${entry_price:.2f}")
        self.logger.info(f"   Stop Loss: ${stop_loss:.2f} (-{self.config['trading']['stop_loss_pct']*100:.1f}%)")
        self.logger.info(f"   Take Profit: ${take_profit:.2f} (+{self.config['trading']['take_profit_pct']*100:.1f}%)")
        self.logger.info(f"   Quantity: {order['quantity']:.4f} AVAX")
        
        return True
    
    def check_exit_conditions(self, analysis):
        """Check if position should be closed"""
        if self.position is None:
            return False, None
        
        current_price = analysis['current_price']
        regime = analysis['regime']
        
        # Update days held
        entry_time = self.position['entry_order']['timestamp']
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        days_held = (datetime.now() - entry_time).days
        self.position['days_held'] = days_held
        
        # Check stop loss
        if current_price <= self.position['stop_loss']:
            return True, "STOP_LOSS"
        
        # Check take profit
        if current_price >= self.position['take_profit']:
            return True, "TAKE_PROFIT"
        
        # Check regime-based exit (your proven logic)
        if regime.should_avoid_trading() and days_held >= 1:
            return True, "REGIME_EXIT"
        
        # Maximum holding period (like your historical trades: 4-6 days)
        if days_held >= 7:
            return True, "MAX_HOLD_PERIOD"
        
        return False, None
    
    def execute_exit_signal(self, reason):
        """Execute exit signal - sell AVAX"""
        if self.position is None:
            self.logger.warning("⚠️ No position to exit")
            return False
        
        current_price = self.get_current_price()
        if current_price is None:
            return False
        
        # Place sell order
        order = self.place_sell_order(self.position['quantity'], current_price)
        if order is None:
            return False
        
        # Calculate trade performance
        entry_price = self.position['entry_price']
        exit_price = order['price']
        trade_return = (exit_price / entry_price) - 1
        trade_pnl = (exit_price - entry_price) * self.position['quantity']
        
        # Create trade record
        trade_record = {
            'entry_time': self.position['entry_order']['timestamp'],
            'exit_time': order['timestamp'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': self.position['quantity'],
            'return_pct': trade_return,
            'pnl_usd': trade_pnl,
            'days_held': self.position['days_held'],
            'exit_reason': reason,
            'entry_analysis': self.position['entry_analysis']
        }
        
        # Update performance metrics
        if trade_return > 0:
            self.performance_metrics['winning_trades'] += 1
        
        self.performance_metrics['total_return'] += trade_return
        self.performance_metrics['current_balance'] += trade_pnl
        
        # Add to trade history
        self.trade_history.append(trade_record)
        
        # Clear position
        self.position = None
        
        # Save state
        self.save_trading_state()
        
        self.logger.info(f"🎯 POSITION CLOSED ({reason}):")
        self.logger.info(f"   Exit: ${exit_price:.2f}")
        self.logger.info(f"   Return: {trade_return*100:+.2f}%")
        self.logger.info(f"   P&L: ${trade_pnl:+.2f}")
        self.logger.info(f"   Days held: {self.position['days_held'] if self.position else 'N/A'}")
        
        return True
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            self.logger.info("🔄 Running trading cycle...")
            
            # Analyze market conditions
            analysis = self.analyze_market_conditions()
            if analysis is None:
                self.logger.error("❌ Market analysis failed")
                return False
            
            current_price = analysis['current_price']
            regime = analysis['regime']
            
            self.logger.info(f"📊 Market Analysis:")
            self.logger.info(f"   Price: ${current_price:.2f}")
            self.logger.info(f"   Regime: {regime.trend.value}/{regime.volatility.value}/{regime.momentum.value}")
            self.logger.info(f"   Confidence: {regime.confidence:.3f}")
            self.logger.info(f"   Buy Level: ${analysis['buy_level']:.2f}")
            self.logger.info(f"   Sell Level: ${analysis['sell_level']:.2f}")
            self.logger.info(f"   Support: ${analysis['support']:.2f}")
            self.logger.info(f"   Resistance: ${analysis['resistance']:.2f}")
            
            self.logger.info(f"   Price Position: {analysis['price_position']:.1%} of range")
            
            # Check if we have a position
            if self.position is not None:
                # Check exit conditions
                should_exit, exit_reason = self.check_exit_conditions(analysis)
                if should_exit:
                    self.execute_exit_signal(exit_reason)
                else:
                    unrealized_pnl = (current_price - self.position['entry_price']) / self.position['entry_price']
                    self.logger.info(f"📈 Position status: {unrealized_pnl*100:+.2f}% (Day {self.position['days_held']})")
            
            else:
                # Check entry conditions
                if analysis['should_enter']:
                    self.logger.info("🟢 BUY SIGNAL DETECTED!")
                    self.execute_entry_signal(analysis)
                else:
                    reasons = []
                    if not analysis['confidence_ok']:
                        reasons.append(f"confidence {regime.confidence:.3f} < 0.60")
                    if not analysis['momentum_filter']:
                        reasons.append("momentum filter")
                    if not analysis['trend_filter']:
                        reasons.append("trend filter")
                    if not analysis['basic_entry']:
                        reasons.append(f"price ${current_price:.2f} < buy level ${analysis['buy_level']:.2f}")
                    if regime.should_avoid_trading():
                        reasons.append("high-risk regime")
                    
                    self.logger.info(f"🔴 No entry signal: {', '.join(reasons)}")
                    
                    # Show what needs to improve
                    improvements = []
                    if not analysis['confidence_ok']:
                        needed = 0.60 - regime.confidence
                        improvements.append(f"confidence +{needed:.3f}")
                    if not analysis['basic_entry']:
                        if current_price < analysis['buy_level']:
                            needed = analysis['buy_level'] - current_price
                            improvements.append(f"price rise ${needed:.2f}")
                        else:
                            needed = current_price - analysis['buy_level']
                            improvements.append(f"price drop ${needed:.2f}")
                    
                    if improvements:
                        self.logger.info(f"   💡 Needs: {', '.join(improvements)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Trading cycle failed: {e}")
            return False
    
    def print_performance_summary(self):
        """Print current performance summary"""
        metrics = self.performance_metrics
        
        print(f"\n📊 PAPER TRADING PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"💰 Current Balance: ${metrics['current_balance']:,.2f}")
        print(f"📈 Total Return: {metrics['total_return']*100:+.2f}%")
        print(f"🎯 Total Trades: {metrics['total_trades']}")
        
        if metrics['total_trades'] > 0:
            win_rate = metrics['winning_trades'] / metrics['total_trades']
            print(f"✅ Win Rate: {win_rate*100:.1f}%")
        
        if self.position:
            current_price = self.get_current_price()
            if current_price:
                unrealized = (current_price - self.position['entry_price']) / self.position['entry_price']
                print(f"📊 Open Position: {unrealized*100:+.2f}% (Day {self.position['days_held']})")
        
        print(f"📅 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_continuous(self, check_interval_minutes=30):
        """Run continuous paper trading"""
        self.logger.info(f"🚀 Starting continuous paper trading (checking every {check_interval_minutes} minutes)")
        
        try:
            while True:
                # Run trading cycle
                success = self.run_trading_cycle()
                
                if success:
                    # Print performance summary
                    self.print_performance_summary()
                
                # Sleep until next check
                self.logger.info(f"💤 Sleeping {check_interval_minutes} minutes...")
                time.sleep(check_interval_minutes * 60)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Paper trading stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Continuous trading failed: {e}")

def main():
    """Main paper trading interface"""
    print("🚀 AVAX Paper Trading Engine")
    print("=" * 35)
    
    engine = PaperTradingEngine()
    
    while True:
        print(f"\n🎯 PAPER TRADING OPTIONS:")
        print("1. Single trading cycle")
        print("2. Market analysis only")
        print("3. Performance summary")
        print("4. Trade history")
        print("5. Start continuous trading")
        print("6. Exit")
        
        choice = input("\nChoose option (1-6): ").strip()
        
        if choice == "1":
            engine.run_trading_cycle()
        
        elif choice == "2":
            analysis = engine.analyze_market_conditions()
            if analysis:
                regime = analysis['regime']
                print(f"\n📊 Current Analysis:")
                print(f"   Price: ${analysis['current_price']:.2f}")
                print(f"   Regime: {regime.trend.value}/{regime.volatility.value}/{regime.momentum.value}")
                print(f"   Confidence: {regime.confidence:.3f}")
                print(f"   Buy Level: ${analysis['buy_level']:.2f}")
                print(f"   Entry Signal: {'🟢 YES' if analysis['should_enter'] else '🔴 NO'}")
        
        elif choice == "3":
            engine.print_performance_summary()
        
        elif choice == "4":
            if engine.trade_history:
                print(f"\n📊 TRADE HISTORY ({len(engine.trade_history)} trades):")
                print("-" * 80)
                for i, trade in enumerate(engine.trade_history[-10:], 1):  # Last 10 trades
                    entry_time = trade['entry_time']
                    if isinstance(entry_time, str):
                        entry_time = datetime.fromisoformat(entry_time).strftime('%m/%d')
                    print(f"{i:2}. {entry_time} ${trade['entry_price']:.2f}→${trade['exit_price']:.2f} "
                          f"{trade['return_pct']*100:+5.1f}% ({trade['days_held']}d) {trade['exit_reason']}")
            else:
                print("📊 No trades executed yet")
        
        elif choice == "5":
            interval = input("Check interval in minutes (30): ").strip()
            interval = int(interval) if interval.isdigit() else 30
            engine.run_continuous(interval)
        
        elif choice == "6":
            print("👋 Paper trading engine closed")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()