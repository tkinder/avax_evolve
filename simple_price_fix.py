# simple_price_fix.py
# Simple fix to add price display to your paper trading engine

def fix_price_display():
    """
    Fix the paper trading engine to display price levels correctly
    """
    try:
        with open('paper_trading_engine.py', 'r') as f:
            content = f.read()
        
        # Find the analyze_market_conditions method and fix the return section
        old_analysis_block = '''            analysis = {
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
            
            return analysis'''
        
        new_analysis_block = '''            # Calculate price position for display
            price_range = max(resistance - support, current_price * 0.01)
            price_position = max(0, min(1, (current_price - support) / price_range))
            
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
            
            self.logger.info(f"📊 Market Analysis:")
            self.logger.info(f"   Price: ${current_price:.2f}")
            self.logger.info(f"   Regime: {regime.trend.value}/{regime.volatility.value}/{regime.momentum.value}")
            self.logger.info(f"   Confidence: {regime.confidence:.3f}")
            self.logger.info(f"   Buy Level: ${buy_level:.2f}")
            self.logger.info(f"   Sell Level: ${sell_level:.2f}")
            self.logger.info(f"   Support: ${support:.2f}")
            self.logger.info(f"   Resistance: ${resistance:.2f}")
            self.logger.info(f"   Price Position: {price_position:.1%} of range")
            
            return analysis'''
        
        if old_analysis_block in content:
            content = content.replace(old_analysis_block, new_analysis_block)
            print("✅ Fixed analysis block with price level displays")
        else:
            print("⚠️ Analysis block pattern not found, trying alternative...")
            # Try a more specific pattern
            if "'price_position': price_position" in content and "return analysis" in content:
                # Find and replace the return analysis section
                import re
                pattern = r"('price_position': price_position\s*}\s*return analysis)"
                
                replacement = '''\\1
            
            self.logger.info(f"📊 Market Analysis:")
            self.logger.info(f"   Price: ${current_price:.2f}")
            self.logger.info(f"   Regime: {regime.trend.value}/{regime.volatility.value}/{regime.momentum.value}")
            self.logger.info(f"   Confidence: {regime.confidence:.3f}")
            self.logger.info(f"   Buy Level: ${buy_level:.2f}")
            self.logger.info(f"   Sell Level: ${sell_level:.2f}")
            self.logger.info(f"   Support: ${support:.2f}")
            self.logger.info(f"   Resistance: ${resistance:.2f}")
            self.logger.info(f"   Price Position: {price_position:.1%} of range")'''
                
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
                print("✅ Applied alternative fix")
        
        # Also fix the error message to show specific prices
        if 'reasons.append("price vs buy level")' in content:
            content = content.replace(
                'reasons.append("price vs buy level")',
                'reasons.append(f"price ${current_price:.2f} > buy level ${buy_level:.2f}")'
            )
            print("✅ Fixed error messages to show specific prices")
        
        # Write the updated file
        with open('paper_trading_engine.py', 'w') as f:
            f.write(content)
        
        print("\n🚀 Paper Trading Engine updated!")
        print("The price levels should now display correctly.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Let me create a manual fix for you...")
        
        # Create a manual instruction file
        with open('manual_fix_instructions.txt', 'w') as f:
            f.write("""
Manual Fix Instructions for paper_trading_engine.py:

Find this section in your analyze_market_conditions method:

            return analysis
            
And replace it with:

            # Add logging before return
            self.logger.info(f"   Buy Level: ${buy_level:.2f}")
            self.logger.info(f"   Sell Level: ${sell_level:.2f}")
            self.logger.info(f"   Support: ${support:.2f}")
            self.logger.info(f"   Resistance: ${resistance:.2f}")
            
            return analysis

This will display the price levels in your trading output.
""")
        print("📝 Created manual_fix_instructions.txt with step-by-step instructions")

if __name__ == "__main__":
    fix_price_display()