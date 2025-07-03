#!/usr/bin/env python3
"""
Quick script to check Binance Testnet balance across different assets
"""

import os
import sys
from binance.client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_testnet_balance():
    """Check testnet balance for all assets"""
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
        
        print("🔍 BINANCE TESTNET BALANCE CHECK")
        print("=" * 40)
        
        # Get account info
        account = client.get_account()
        
        # Filter out zero balances and show significant ones
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
        
        if significant_balances:
            print(f"📊 Non-zero balances ({len(significant_balances)} assets):")
            print("-" * 40)
            for bal in significant_balances:
                print(f"   {bal['asset']:6} | Free: {bal['free']:>12.8f} | Locked: {bal['locked']:>12.8f} | Total: {bal['total']:>12.8f}")
        else:
            print("📊 No balances found")
        
        print("\n🎯 KEY ASSETS FOR TRADING:")
        print("-" * 40)
        key_assets = ['USDT', 'BNB', 'BTC', 'ETH', 'AVAX', 'SOL']
        
        for asset in key_assets:
            bal = next((b for b in significant_balances if b['asset'] == asset), None)
            if bal:
                print(f"   {asset:6} | {bal['total']:>15.8f}")
            else:
                print(f"   {asset:6} | {0:>15.8f}")
        
        print("\n💰 TRADING IMPLICATIONS:")
        print("-" * 40)
        
        # Check USDT balance (main trading currency)
        usdt_balance = next((b['total'] for b in significant_balances if b['asset'] == 'USDT'), 0)
        
        if usdt_balance > 0:
            print(f"✅ USDT Balance: ${usdt_balance:,.2f}")
            print(f"   - This appears to be your total trading capital")
            if usdt_balance >= 10000:
                print(f"   - Could allocate ~$3,333 per asset (AVAX/BTC/SOL)")
            elif usdt_balance >= 1000:
                print(f"   - Could allocate ~${usdt_balance/3:,.0f} per asset")
            else:
                print(f"   - Limited capital, consider single asset focus")
        else:
            print("❌ No USDT balance found for trading")
        
        # Check if we have actual crypto assets
        crypto_assets = [b for b in significant_balances if b['asset'] in ['BTC', 'ETH', 'AVAX', 'SOL', 'BNB']]
        if crypto_assets:
            print(f"\n🪙 CRYPTO ASSETS AVAILABLE:")
            for asset in crypto_assets:
                print(f"   - {asset['asset']}: {asset['total']:.8f}")
        
        print(f"\n📝 CONCLUSION:")
        if usdt_balance >= 10000:
            print("   ✅ Sufficient balance for multi-asset strategy")
            print("   💡 Recommendation: Implement allocation strategy")
        elif usdt_balance >= 1000:
            print("   ⚠️  Limited balance - consider sequential deployment")
        else:
            print("   ❌ Insufficient balance for multi-asset trading")
            print("   💡 Focus on single asset or request testnet reset")
            
    except Exception as e:
        print(f"❌ Error checking balance: {e}")
        print("\n💡 Possible solutions:")
        print("   1. Check API credentials in .env file")
        print("   2. Verify testnet API keys are correct")
        print("   3. Check internet connection")

if __name__ == "__main__":
    check_testnet_balance()
