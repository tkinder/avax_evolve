# direct_fix.py - Apply this fix to your paper_trading_engine.py

# Find this section in your run_trading_cycle method (around line 520):

# REPLACE THIS:
"""
            self.logger.info(f"📊 Market Analysis:")
            self.logger.info(f"   Price: ${current_price:.2f}")
            self.logger.info(f"   Regime: {regime.trend.value}/{regime.volatility.value}/{regime.momentum.value}")
            self.logger.info(f"   Confidence: {regime.confidence:.3f}")
            self.logger.info(f"   Buy Level: ${buy_level:.2f}")
            self.logger.info(f"   Sell Level: ${sell_level:.2f}")
            self.logger.info(f"   Support: ${support:.2f}")
            self.logger.info(f"   Resistance: ${resistance:.2f}")
            
            # Calculate and show price position
            price_range = max(resistance - support, current_price * 0.01)
            price_position = max(0, min(1, (current_price - support) / price_range))
            self.logger.info(f"   Price Position: {price_position:.1%} of range")
"""

# WITH THIS:
"""
            self.logger.info(f"📊 Market Analysis:")
            self.logger.info(f"   Price: ${current_price:.2f}")
            self.logger.info(f"   Regime: {regime.trend.value}/{regime.volatility.value}/{regime.momentum.value}")
            self.logger.info(f"   Confidence: {regime.confidence:.3f}")
            self.logger.info(f"   Buy Level: ${analysis['buy_level']:.2f}")
            self.logger.info(f"   Sell Level: ${analysis['sell_level']:.2f}")
            self.logger.info(f"   Support: ${analysis['support']:.2f}")
            self.logger.info(f"   Resistance: ${analysis['resistance']:.2f}")
            self.logger.info(f"   Price Position: {analysis['price_position']:.1%} of range")
"""

print("""
🔧 MANUAL FIX INSTRUCTIONS:

1. Open your paper_trading_engine.py file
2. Find this line (around line 520):
   self.logger.info(f"   Buy Level: ${buy_level:.2f}")

3. Replace the logging section with:
   self.logger.info(f"   Buy Level: ${analysis['buy_level']:.2f}")
   self.logger.info(f"   Sell Level: ${analysis['sell_level']:.2f}")
   self.logger.info(f"   Support: ${analysis['support']:.2f}")
   self.logger.info(f"   Resistance: ${analysis['resistance']:.2f}")
   self.logger.info(f"   Price Position: {analysis['price_position']:.1%} of range")

4. Also fix the error message section (around line 550):
   Change: reasons.append(f"price ${current_price:.2f} > buy level ${buy_level:.2f}")
   To: reasons.append(f"price ${current_price:.2f} > buy level ${analysis['buy_level']:.2f}")

5. Fix the improvement section (around line 560):
   Change: needed = current_price - buy_level
   To: needed = current_price - analysis['buy_level']

This will use the values from the analysis dictionary instead of undefined variables.
""")

# Automatic fix function
def apply_fix():
    try:
        with open('paper_trading_engine.py', 'r') as f:
            content = f.read()
        
        # Fix 1: Replace buy_level with analysis['buy_level'] in logging
        content = content.replace('${buy_level:.2f}', "${analysis['buy_level']:.2f}")
        content = content.replace('${sell_level:.2f}', "${analysis['sell_level']:.2f}")
        content = content.replace('${support:.2f}', "${analysis['support']:.2f}")
        content = content.replace('${resistance:.2f}', "${analysis['resistance']:.2f}")
        
        # Fix 2: Replace price_position calculation
        old_calc = """# Calculate and show price position
            price_range = max(resistance - support, current_price * 0.01)
            price_position = max(0, min(1, (current_price - support) / price_range))
            self.logger.info(f"   Price Position: {price_position:.1%} of range")"""
        
        new_calc = """self.logger.info(f"   Price Position: {analysis['price_position']:.1%} of range")"""
        
        content = content.replace(old_calc, new_calc)
        
        # Fix 3: Fix error message
        content = content.replace(
            'f"price ${current_price:.2f} > buy level ${buy_level:.2f}"',
            'f"price ${current_price:.2f} > buy level ${analysis[\'buy_level\']:.2f}"'
        )
        
        # Fix 4: Fix improvement calculation
        content = content.replace(
            'needed = current_price - buy_level',
            'needed = current_price - analysis[\'buy_level\']'
        )
        
        # Write the fixed file
        with open('paper_trading_engine.py', 'w') as f:
            f.write(content)
        
        print("✅ Successfully applied automatic fix!")
        print("Now run: python paper_trading_engine.py")
        
    except Exception as e:
        print(f"❌ Automatic fix failed: {e}")
        print("Please apply the manual fix shown above.")

if __name__ == "__main__":
    apply_fix()