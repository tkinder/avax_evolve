# quick_price_display_fix.py
# Run this to add price level display to your paper trading engine

import re

def fix_paper_trading_engine():
    """
    Add price level displays to paper_trading_engine.py
    """
    try:
        # Read the current file
        with open('paper_trading_engine.py', 'r') as f:
            content = f.read()
        
        # Find the market analysis logging section and enhance it
        old_pattern = r'(self\.logger\.info\(f"   Confidence: {regime\.confidence:.3f}"\))'
        
        new_code = '''self.logger.info(f"   Confidence: {regime.confidence:.3f}")
            self.logger.info(f"   Buy Level: ${buy_level:.2f}")
            self.logger.info(f"   Sell Level: ${sell_level:.2f}")
            self.logger.info(f"   Support: ${support:.2f}")
            self.logger.info(f"   Resistance: ${resistance:.2f}")
            self.logger.info(f"   Price Position: {price_position:.1%} of range")'''
        
        if re.search(old_pattern, content):
            content = re.sub(old_pattern, new_code, content)
            print("✅ Enhanced market analysis logging")
        else:
            print("⚠️ Market analysis pattern not found")
        
        # Enhance the error message to show specific buy level
        old_error_pattern = r'reasons\.append\("price vs buy level"\)'
        new_error_code = 'reasons.append(f"price ${current_price:.2f} > buy level ${buy_level:.2f}")'
        
        if 'reasons.append("price vs buy level")' in content:
            content = content.replace('reasons.append("price vs buy level")', new_error_code)
            print("✅ Enhanced error messages with specific prices")
        
        # Add improvement suggestions
        improvement_code = '''
                    # Show what needs to improve
                    improvements = []
                    if not analysis['confidence_ok']:
                        needed = 0.60 - regime.confidence
                        improvements.append(f"confidence +{needed:.3f}")
                    if not analysis['basic_entry']:
                        needed = current_price - buy_level
                        improvements.append(f"price drop ${abs(needed):.2f}")
                    
                    if improvements:
                        self.logger.info(f"   💡 Needs: {', '.join(improvements)}")'''
        
        # Find the location to add improvement suggestions
        if 'self.logger.info(f"🔴 No entry signal:' in content and 'improvements' not in content:
            pattern = r'(self\.logger\.info\(f"🔴 No entry signal: {.*?}"\))'
            replacement = r'\1' + improvement_code
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            print("✅ Added improvement suggestions")
        
        # Write the updated file
        with open('paper_trading_engine.py', 'w') as f:
            f.write(content)
        
        print("\n🚀 Paper Trading Engine updated successfully!")
        print("Now run: python paper_trading_engine.py")
        
    except FileNotFoundError:
        print("❌ paper_trading_engine.py not found")
    except Exception as e:
        print(f"❌ Error updating file: {e}")

if __name__ == "__main__":
    fix_paper_trading_engine()