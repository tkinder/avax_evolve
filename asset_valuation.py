#!/usr/bin/env python3
"""
Testnet Asset Valuation Script
Calculates the total USDT value of liquidatable crypto assets
"""

import os
import sys
from binance.client import Client
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def calculate_liquidation_value():
    """Calculate total USDT value of liquidatable assets"""
    try:
        # Initialize Binance testnet client
        api_key = os.getenv('BINANCE_TESTNET_API_KEY')
        api_secret = os.getenv('BINANCE_TESTNET_SECRET')
        
        if not api_key or not api_secret:
            print("❌ Missing API credentials in .env file")
            return None
        
        client = Client(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True
        )
        
        print("💎 TESTNET ASSET VALUATION REPORT")
        print("=" * 60)
        
        # Assets we want to keep (don't liquidate)
        keep_assets = {'USDT', 'BNB', 'BTC', 'ETH', 'AVAX', 'SOL'}
        
        # Get account balances
        account = client.get_account()
        
        # Get all available trading pairs for price lookup
        exchange_info = client.get_exchange_info()
        usdt_pairs = set()
        for symbol_info in exchange_info['symbols']:
            if (symbol_info['quoteAsset'] == 'USDT' and 
                symbol_info['status'] == 'TRADING'):
                usdt_pairs.add(symbol_info['baseAsset'])
        
        print(f"📊 Available USDT trading pairs: {len(usdt_pairs)}")
        
        # Analyze balances
        keep_balances = []
        liquidatable_balances = []
        unpaired_balances = []
        
        total_liquidation_value = 0.0
        successful_valuations = 0
        failed_valuations = 0
        
        print(f"\n🔍 ASSET ANALYSIS:")
        print("-" * 60)
        
        for balance in account['balances']:
            free_balance = float(balance['free'])
            locked_balance = float(balance['locked'])
            total_balance = free_balance + locked_balance
            
            if total_balance <= 0:
                continue
                
            asset = balance['asset']
            
            if asset in keep_assets:
                keep_balances.append({
                    'asset': asset,
                    'amount': total_balance,
                    'status': 'KEEP'
                })
                print(f"✅ KEEP     | {asset:8} | {total_balance:>18.8f}")
                
            elif asset in usdt_pairs:
                # Try to get current price
                try:
                    ticker = client.get_symbol_ticker(symbol=f"{asset}USDT")
                    price = float(ticker['price'])
                    usdt_value = total_balance * price
                    total_liquidation_value += usdt_value
                    successful_valuations += 1
                    
                    liquidatable_balances.append({
                        'asset': asset,
                        'amount': total_balance,
                        'price': price,
                        'usdt_value': usdt_value,
                        'status': 'LIQUIDATE'
                    })
                    
                    print(f"💰 SELL     | {asset:8} | {total_balance:>18.8f} | ${price:>8.4f} | ${usdt_value:>10.2f}")
                    
                    # Add small delay to avoid rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    failed_valuations += 1
                    liquidatable_balances.append({
                        'asset': asset,
                        'amount': total_balance,
                        'price': None,
                        'usdt_value': 0,
                        'status': 'LIQUIDATE_NO_PRICE'
                    })
                    print(f"⚠️  SELL     | {asset:8} | {total_balance:>18.8f} | Price Error: {str(e)[:20]}...")
                    
            else:
                unpaired_balances.append({
                    'asset': asset,
                    'amount': total_balance,
                    'status': 'NO_USDT_PAIR'
                })
                print(f"❌ SKIP     | {asset:8} | {total_balance:>18.8f} | No USDT pair")
        
        print("\n" + "=" * 60)
        print("📈 LIQUIDATION SUMMARY:")
        print("-" * 60)
        
        current_usdt = next((b['amount'] for b in keep_balances if b['asset'] == 'USDT'), 0)
        
        print(f"💵 Current USDT Balance:           ${current_usdt:>12,.2f}")
        print(f"💰 Total Liquidation Value:       ${total_liquidation_value:>12,.2f}")
        print(f"🎯 Potential Total USDT:          ${current_usdt + total_liquidation_value:>12,.2f}")
        print(f"📊 Assets to Keep:                {len(keep_balances):>12}")
        print(f"💎 Assets Available to Liquidate: {len(liquidatable_balances):>12}")
        print(f"✅ Successful Price Lookups:      {successful_valuations:>12}")
        print(f"❌ Failed Price Lookups:          {failed_valuations:>12}")
        print(f"🚫 No USDT Pairs:                 {len(unpaired_balances):>12}")
        
        print(f"\n🎯 TARGET ANALYSIS:")
        print("-" * 30)
        target_additional = 20000  # $20K additional USDT
        target_total = current_usdt + target_additional
        
        if total_liquidation_value >= target_additional:
            print(f"✅ GOAL ACHIEVABLE!")
            print(f"   Target Additional: ${target_additional:,}")
            print(f"   Available Value:   ${total_liquidation_value:,.2f}")
            print(f"   Excess Available:  ${total_liquidation_value - target_additional:,.2f}")
            
            # Calculate how much we need to liquidate
            percentage_needed = (target_additional / total_liquidation_value) * 100
            print(f"   Need to liquidate: {percentage_needed:.1f}% of available assets")
            
        else:
            print(f"⚠️  TARGET CHALLENGING")
            print(f"   Target Additional: ${target_additional:,}")
            print(f"   Available Value:   ${total_liquidation_value:,.2f}")
            print(f"   Shortfall:         ${target_additional - total_liquidation_value:,.2f}")
            print(f"   Can achieve:       ${current_usdt + total_liquidation_value:,.2f} total USDT")
        
        print(f"\n📋 TOP LIQUIDATION CANDIDATES:")
        print("-" * 45)
        
        # Sort by USDT value descending
        sorted_assets = sorted([a for a in liquidatable_balances if a['usdt_value'] > 0], 
                              key=lambda x: x['usdt_value'], reverse=True)
        
        for i, asset_info in enumerate(sorted_assets[:15], 1):  # Top 15
            print(f"{i:2}. {asset_info['asset']:8} | "
                  f"{asset_info['amount']:>15.8f} | "
                  f"${asset_info['usdt_value']:>10.2f}")
        
        if len(sorted_assets) > 15:
            remaining_value = sum(a['usdt_value'] for a in sorted_assets[15:])
            print(f"    ... and {len(sorted_assets) - 15} more worth ${remaining_value:.2f}")
        
        print(f"\n💡 LIQUIDATION STRATEGY RECOMMENDATION:")
        print("-" * 45)
        
        daily_limit = 5000  # $5K per day as requested
        days_needed = target_additional / daily_limit
        
        print(f"Daily Liquidation Limit: ${daily_limit:,}")
        print(f"Days to Reach Target:    {days_needed:.1f} days")
        print(f"Assets per Day:          ~{len(sorted_assets) / days_needed:.0f} assets")
        
        # Save detailed report
        report = {
            'current_usdt': current_usdt,
            'total_liquidation_value': total_liquidation_value,
            'target_additional': target_additional,
            'keep_balances': keep_balances,
            'liquidatable_balances': sorted_assets,
            'daily_limit': daily_limit,
            'days_needed': days_needed
        }
        
        return report
        
    except Exception as e:
        print(f"❌ Error calculating liquidation value: {e}")
        return None

if __name__ == "__main__":
    print("Starting asset valuation analysis...")
    report = calculate_liquidation_value()
    
    if report:
        print(f"\n🚀 Ready to proceed with liquidation strategy!")
        print(f"   Total available: ${report['total_liquidation_value']:,.2f}")
        print(f"   Target: ${report['target_additional']:,} additional USDT")
        print(f"   Timeline: {report['days_needed']:.1f} days at ${report['daily_limit']:,}/day")
    else:
        print(f"\n❌ Could not complete valuation analysis")
