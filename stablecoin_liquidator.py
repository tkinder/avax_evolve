#!/usr/bin/env python3
"""
Stablecoin Liquidation Script
Converts USDC, FDUSD, and TUSD to USDT over 4-5 days
Simple, safe, and effective capital optimization
"""

import os
import sys
import json
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class StablecoinLiquidator:
    """
    Liquidates stablecoins to USDT for increased trading capital
    """
    
    def __init__(self, config_file='liquidation_config.json'):
        """Initialize the stablecoin liquidator"""
        self.config = self.load_config(config_file)
        self.setup_logging()
        
        # Initialize Binance testnet client
        self.client = Client(
            api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
            api_secret=os.getenv('BINANCE_TESTNET_SECRET'),
            testnet=True
        )
        
        # Liquidation targets (in order of execution)
        self.liquidation_schedule = [
            {'asset': 'USDC', 'symbol': 'USDCUSDT', 'day': 1},
            {'asset': 'FDUSD', 'symbol': 'FDUSDUSDT', 'day': 2}, 
            {'asset': 'TUSD', 'symbol': 'TUSDUSDT', 'day': 3}
        ]
        
        # Load existing state
        self.load_liquidation_state()
        
        print("💰 Stablecoin Liquidator initialized")
        print(f"🎯 Target: Convert USDC, FDUSD, TUSD → USDT")
        
    def load_config(self, config_file):
        """Load liquidation configuration"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Create default config
        default_config = {
            "liquidation": {
                "daily_limit_usd": 15000,  # Per day limit
                "min_amount_threshold": 1.0,  # Minimum amount to liquidate
                "safety_buffer_pct": 0.01,  # Keep 1% as buffer
                "dry_run": False  # Set to True for testing
            },
            "logging": {
                "log_file": "stablecoin_liquidation.log",
                "detailed_logging": True
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        print(f"📝 Created default config: {config_file}")
        return default_config
    
    def setup_logging(self):
        """Setup logging for liquidation tracking"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['logging']['log_file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_liquidation_state(self):
        """Load existing liquidation state"""
        state_file = 'liquidation_state.json'
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    self.state = json.load(f)
                self.logger.info("📊 Loaded existing liquidation state")
            except Exception as e:
                self.logger.error(f"❌ Failed to load liquidation state: {e}")
                self.state = self.create_fresh_state()
        else:
            self.state = self.create_fresh_state()
    
    def create_fresh_state(self):
        """Create fresh liquidation state"""
        return {
            'start_date': datetime.now().isoformat(),
            'completed_liquidations': [],
            'total_usdt_gained': 0.0,
            'current_day': 0,
            'status': 'READY'
        }
    
    def save_liquidation_state(self):
        """Save current liquidation state"""
        with open('liquidation_state.json', 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_account_balance(self, asset):
        """Get current balance for specific asset"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            return 0.0
        except Exception as e:
            self.logger.error(f"❌ Failed to get {asset} balance: {e}")
            return 0.0
    
    def get_current_price(self, symbol):
        """Get current price for trading pair"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            self.logger.error(f"❌ Failed to get price for {symbol}: {e}")
            return None
    
    def calculate_liquidation_amount(self, asset, current_balance):
        """Calculate how much to liquidate (keeping small buffer)"""
        if current_balance <= self.config['liquidation']['min_amount_threshold']:
            return 0.0
        
        # Keep small buffer (1% by default)
        buffer_amount = current_balance * self.config['liquidation']['safety_buffer_pct']
        liquidation_amount = current_balance - buffer_amount
        
        return max(0.0, liquidation_amount)
    
    def execute_liquidation(self, asset_info):
        """Execute liquidation for a specific asset"""
        asset = asset_info['asset']
        symbol = asset_info['symbol']
        
        self.logger.info(f"🔄 Starting liquidation: {asset} → USDT")
        
        # Check current balance
        current_balance = self.get_account_balance(asset)
        if current_balance <= 0:
            self.logger.warning(f"⚠️ No {asset} balance to liquidate")
            return False
        
        # Calculate liquidation amount
        liquidation_amount = self.calculate_liquidation_amount(asset, current_balance)
        if liquidation_amount <= 0:
            self.logger.warning(f"⚠️ {asset} amount too small to liquidate")
            return False
        
        # Get current price
        current_price = self.get_current_price(symbol)
        if current_price is None:
            self.logger.error(f"❌ Could not get price for {symbol}")
            return False
        
        estimated_usdt = liquidation_amount * current_price
        
        self.logger.info(f"💰 {asset} Liquidation Plan:")
        self.logger.info(f"   Current Balance: {current_balance:.8f} {asset}")
        self.logger.info(f"   Liquidation Amount: {liquidation_amount:.8f} {asset}")
        self.logger.info(f"   Current Price: ${current_price:.4f}")
        self.logger.info(f"   Estimated USDT: ${estimated_usdt:.2f}")
        
        # Dry run check
        if self.config['liquidation']['dry_run']:
            self.logger.info("🧪 DRY RUN MODE - No actual trade executed")
            return True
        
        # Get USDT balance before trade
        usdt_before = self.get_account_balance('USDT')
        
        # Execute market sell order
        try:
            self.logger.info(f"🔴 Placing SELL order: {liquidation_amount:.8f} {asset}")
            
            # Place market sell order
            order = self.client.order_market_sell(
                symbol=symbol,
                quantity=f"{liquidation_amount:.8f}"
            )
            
            # Get actual fill details
            fill_price = float(order['fills'][0]['price']) if order['fills'] else current_price
            fill_quantity = float(order['executedQty'])
            actual_usdt_received = sum(float(fill['qty']) * float(fill['price']) for fill in order['fills'])
            
            # Get USDT balance after trade
            time.sleep(1)  # Brief pause for balance update
            usdt_after = self.get_account_balance('USDT')
            actual_usdt_gain = usdt_after - usdt_before
            
            self.logger.info(f"✅ {asset} liquidation completed!")
            self.logger.info(f"   Executed: {fill_quantity:.8f} {asset}")
            self.logger.info(f"   Fill Price: ${fill_price:.4f}")
            self.logger.info(f"   USDT Received: ${actual_usdt_received:.2f}")
            self.logger.info(f"   Actual Balance Increase: ${actual_usdt_gain:.2f}")
            
            # Record the liquidation
            liquidation_record = {
                'timestamp': datetime.now().isoformat(),
                'asset': asset,
                'symbol': symbol,
                'quantity_sold': fill_quantity,
                'fill_price': fill_price,
                'usdt_received': actual_usdt_received,
                'balance_increase': actual_usdt_gain,
                'order_id': order['orderId']
            }
            
            self.state['completed_liquidations'].append(liquidation_record)
            self.state['total_usdt_gained'] += actual_usdt_gain
            self.save_liquidation_state()
            
            return True
            
        except BinanceAPIException as e:
            self.logger.error(f"❌ Binance API error during {asset} liquidation: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during {asset} liquidation: {e}")
            return False
    
    def get_liquidation_status(self):
        """Get current liquidation progress"""
        completed = len(self.state['completed_liquidations'])
        total = len(self.liquidation_schedule)
        
        print(f"\n📊 LIQUIDATION PROGRESS")
        print("=" * 40)
        print(f"💰 Total USDT Gained: ${self.state['total_usdt_gained']:,.2f}")
        print(f"✅ Completed: {completed}/{total} assets")
        print(f"📅 Start Date: {self.state['start_date'][:10]}")
        
        if self.state['completed_liquidations']:
            print(f"\n📋 COMPLETED LIQUIDATIONS:")
            for i, record in enumerate(self.state['completed_liquidations'], 1):
                timestamp = record['timestamp'][:10]  # Date only
                print(f"  {i}. {record['asset']:6} | ${record['usdt_received']:>8.2f} | {timestamp}")
        
        # Show remaining assets
        completed_assets = {record['asset'] for record in self.state['completed_liquidations']}
        remaining_assets = [item for item in self.liquidation_schedule 
                          if item['asset'] not in completed_assets]
        
        if remaining_assets:
            print(f"\n⏳ REMAINING ASSETS:")
            for item in remaining_assets:
                balance = self.get_account_balance(item['asset'])
                print(f"  📅 Day {item['day']}: {item['asset']:6} | Balance: {balance:>12.2f}")
        
        current_usdt = self.get_account_balance('USDT')
        print(f"\n💵 Current USDT Balance: ${current_usdt:,.2f}")
        
        if completed == total:
            print(f"🎉 ALL LIQUIDATIONS COMPLETE!")
            print(f"🚀 Ready for multi-asset trading with ${current_usdt:,.2f}")
    
    def run_daily_liquidation(self):
        """Run today's liquidation if any is scheduled"""
        # Find next asset to liquidate
        completed_assets = {record['asset'] for record in self.state['completed_liquidations']}
        next_asset = None
        
        for item in self.liquidation_schedule:
            if item['asset'] not in completed_assets:
                next_asset = item
                break
        
        if next_asset is None:
            self.logger.info("🎉 All liquidations completed!")
            self.get_liquidation_status()
            return True
        
        self.logger.info(f"🎯 Today's target: {next_asset['asset']}")
        
        # Execute liquidation
        success = self.execute_liquidation(next_asset)
        
        if success:
            self.logger.info(f"✅ Day {next_asset['day']} liquidation successful!")
        else:
            self.logger.error(f"❌ Day {next_asset['day']} liquidation failed!")
        
        # Show updated status
        self.get_liquidation_status()
        
        return success
    
    def run_all_liquidations(self):
        """Run all liquidations in sequence (for testing or one-time execution)"""
        self.logger.info("🚀 Starting complete liquidation sequence")
        
        for item in self.liquidation_schedule:
            asset = item['asset']
            
            # Skip if already completed
            completed_assets = {record['asset'] for record in self.state['completed_liquidations']}
            if asset in completed_assets:
                self.logger.info(f"✅ {asset} already liquidated, skipping")
                continue
            
            self.logger.info(f"\n🎯 Processing {asset} (Day {item['day']})")
            
            success = self.execute_liquidation(item)
            
            if success:
                self.logger.info(f"✅ {asset} liquidation successful!")
                time.sleep(2)  # Brief pause between liquidations
            else:
                self.logger.error(f"❌ {asset} liquidation failed! Stopping sequence.")
                break
        
        self.get_liquidation_status()

def main():
    """Main liquidation interface"""
    print("💰 STABLECOIN LIQUIDATION SYSTEM")
    print("=" * 40)
    
    liquidator = StablecoinLiquidator()
    
    while True:
        print(f"\n🎯 LIQUIDATION OPTIONS:")
        print("1. Run today's liquidation")
        print("2. View liquidation status") 
        print("3. Run ALL liquidations (testing)")
        print("4. Check current balances")
        print("5. Toggle dry run mode")
        print("6. Exit")
        
        choice = input("\nChoose option (1-6): ").strip()
        
        if choice == "1":
            liquidator.run_daily_liquidation()
        
        elif choice == "2":
            liquidator.get_liquidation_status()
        
        elif choice == "3":
            confirm = input("⚠️ Run ALL liquidations now? (y/N): ").strip().lower()
            if confirm == 'y':
                liquidator.run_all_liquidations()
        
        elif choice == "4":
            print(f"\n💰 CURRENT BALANCES:")
            print("-" * 30)
            assets = ['USDT', 'USDC', 'FDUSD', 'TUSD', 'BTC', 'ETH', 'AVAX', 'SOL']
            for asset in assets:
                balance = liquidator.get_account_balance(asset)
                print(f"  {asset:6} | {balance:>15.8f}")
        
        elif choice == "5":
            current_mode = liquidator.config['liquidation']['dry_run']
            liquidator.config['liquidation']['dry_run'] = not current_mode
            new_mode = "DRY RUN" if liquidator.config['liquidation']['dry_run'] else "LIVE TRADING"
            print(f"🔄 Mode changed to: {new_mode}")
        
        elif choice == "6":
            print("👋 Liquidation system closed")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
