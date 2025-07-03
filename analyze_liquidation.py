#!/usr/bin/env python3
"""
Testnet Asset Liquidation Analyzer
Checks what assets we have and what can be converted to USDT
"""

import os
import sys
from binance.client import Client
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def analyze_liquidation_opportunities():
    """Check what assets can be converted to USDT"""
    try:
        # Initialize Binance testnet client
        api_key = os.getenv('BINANCE_TESTNET_API_KEY')
        api_secret = os.getenv('BINANCE_TESTNET_SECRET')
        
        if not api_key or not api_secret:
            print("❌ Missing API credentials in .env file")
            return
        
        client = Client(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True
        )
        
        print("🏦 TESTNET ASSET LIQUIDATION ANALYSIS")
        print("=" * 50)
        
        # Get account info
        account = client.get_account()
        
        # Get all available trading pairs
        exchange_info = client.get_exchange_info()
        
        # Filter for USDT pairs
        usdt_pairs = []
        for symbol_info in exchange_info['symbols']:
            if (symbol_info['quoteAsset'] == 'USDT' and 
                symbol_info['status'] == 'TRADING'):
                usdt_pairs.append(symbol_info['baseAsset'])
        
        print(f"📊 Available USDT trading pairs: {len(usdt_pairs)}")
        print(f"   {', '.join(sorted(usdt_pairs)[:20])}{'...' if len(usdt_pairs) > 20 else ''}")
        
        # Check current balances
        significant_balances = []
        for balance in account['balances']:
            free_balance = float(balance['free'])
            locked_balance = float(balance['locked'])
            total_balance = free_balance + locked_balance
            
            if total_balance > 0:
                significant_balances.append({
                    'asset': balance['asset'],
                    'free': free_balance,
                    'locked': locked_balance,
                    'total': total_balance
                })
        
        print(f"\n💰 Current Non-Zero Balances: {len(significant_balances)}")
        print("-" * 50)
        
        # Assets we want to keep
        keep_assets = {'USDT', 'BNB', 'BTC', 'ETH', 'AVAX', 'SOL'}
        
        # Assets we could potentially liquidate
        liquidatable_assets = []
        keep_assets_found = []
        
        total_estimated_usdt_value = 0
        
        for bal in significant_balances:
            asset = bal['asset']
            amount = bal['total']
            
            if asset in keep_assets:
                keep_assets_found.append(bal)
                print(f"✅ KEEP  | {asset:6} | {amount:>15.8f}")
            elif asset in usdt_pairs:
                liquidatable_assets.append(bal)
                # Try to get current price for estimation
                try:
                    ticker = client.get_symbol_ticker(symbol=f"{asset}USDT")
                    price = float(ticker['price'])
                    estimated_usdt = amount * price
                    total_estimated_usdt_value += estimated_usdt
                    print(f"🔄 SELL  | {asset:6} | {amount:>15.8f} | ~${estimated_usdt:>10.2f}")
                except:
                    print(f"🔄 SELL  | {asset:6} | {amount:>15.8f} | Price N/A")
            else:
                print(f"❌ SKIP  | {asset:6} | {amount:>15.8f} | No USDT pair")
        
        print("\n" + "=" * 50)
        print("📈 LIQUIDATION POTENTIAL:")
        print("-" * 50)
        
        current_usdt = next((b['total'] for b in keep_assets_found if b['asset'] == 'USDT'), 0)
        potential_total = current_usdt + total_estimated_usdt_value
        
        print(f"💵 Current USDT Balance: ${current_usdt:,.2f}")
        print(f"💰 Potential from liquidation: ~${total_estimated_usdt_value:,.2f}")
        print(f"🎯 Total Potential USDT: ~${potential_total:,.2f}")
        
        if total_estimated_usdt_value > 1000:
            print(f"\n✅ RECOMMENDATION: LIQUIDATE ASSETS!")
            print(f"   - Could increase trading capital by ${total_estimated_usdt_value:,.2f}")
            print(f"   - Would allow larger position sizes")
            print(f"   - {len(liquidatable_assets)} assets available for conversion")
        else:
            print(f"\n⚠️  Limited liquidation value (${total_estimated_usdt_value:,.2f})")
        
        print(f"\n🛠️  IMPLEMENTATION STRATEGY:")
        print("-" * 30)
        
        if liquidatable_assets:
            print("1. Create liquidation script")
            print("2. Market sell each unwanted asset for USDT")
            print("3. Update trading capital allocation")
            print("4. Implement multi-asset strategy with larger capital")
            
            print(f"\n📋 ASSETS TO LIQUIDATE ({len(liquidatable_assets)}):")
            for asset in liquidatable_assets[:10]:  # Show first 10
                print(f"   - {asset['asset']}: {asset['total']:.8f}")
            if len(liquidatable_assets) > 10:
                print(f"   ... and {len(liquidatable_assets) - 10} more")
        else:
            print("No assets available for liquidation")
        
        print(f"\n🔍 Want to see the liquidation script? (Y/n)")
        
        return {
            'liquidatable_assets': liquidatable_assets,
            'potential_usdt_value': total_estimated_usdt_value,
            'current_usdt': current_usdt
        }
            
    except Exception as e:
        print(f"❌ Error analyzing liquidation opportunities: {e}")
        return None

if __name__ == "__main__":
    result = analyze_liquidation_opportunities()
    
    if result and result['liquidatable_assets']:
        print(f"\n💡 Next step: Create automated liquidation script!")
        print(f"   Potential gain: ${result['potential_usdt_value']:,.2f}")
