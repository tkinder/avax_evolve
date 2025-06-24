# fix_price_logic.py
# Fix the price direction logic in paper_trading_engine.py

def fix_price_direction_logic():
    """
    Fix the backwards price direction logic in the paper trading engine
    """
    try:
        with open('paper_trading_engine.py', 'r') as f:
            content = f.read()
        
        # Fix 1: Correct the price comparison logic
        # The basic_entry check is backwards - should be price <= buy_level for entry
        old_comparison = 'f"price ${current_price:.2f} > buy level ${analysis[\'buy_level\']:.2f}"'
        new_comparison = 'f"price ${current_price:.2f} < buy level ${analysis[\'buy_level\']:.2f}"'
        
        if old_comparison in content:
            content = content.replace(old_comparison, new_comparison)
            print("✅ Fixed price comparison logic")
        else:
            print("⚠️ Price comparison pattern not found")
        
        # Fix 2: Correct the improvement suggestion logic
        old_improvement_logic = '''if not analysis['basic_entry']:
                        needed = current_price - analysis['buy_level']
                        improvements.append(f"price drop ${abs(needed):.2f}")'''
        
        new_improvement_logic = '''if not analysis['basic_entry']:
                        if current_price < analysis['buy_level']:
                            needed = analysis['buy_level'] - current_price
                            improvements.append(f"price rise ${needed:.2f}")
                        else:
                            needed = current_price - analysis['buy_level']
                            improvements.append(f"price drop ${needed:.2f}")'''
        
        if 'needed = current_price - analysis[\'buy_level\']' in content:
            content = content.replace(old_improvement_logic, new_improvement_logic)
            print("✅ Fixed improvement suggestion logic")
        else:
            # Try alternative pattern
            alt_pattern = '''needed = current_price - analysis['buy_level']
                        improvements.append(f"price drop ${abs(needed):.2f}")'''
            
            alt_replacement = '''if current_price < analysis['buy_level']:
                            needed = analysis['buy_level'] - current_price
                            improvements.append(f"price rise ${needed:.2f}")
                        else:
                            needed = current_price - analysis['buy_level']
                            improvements.append(f"price drop ${needed:.2f}")'''
            
            if alt_pattern in content:
                content = content.replace(alt_pattern, alt_replacement)
                print("✅ Fixed improvement suggestion logic (alternative pattern)")
            else:
                print("⚠️ Improvement logic pattern not found")
        
        # Write the fixed content
        with open('paper_trading_engine.py', 'w') as f:
            f.write(content)
        
        print("\n🚀 Price direction logic fixed!")
        print("Now your system will correctly say:")
        print("  - 'price $17.16 < buy level $18.03' ✅")
        print("  - 'price rise $0.87' ✅")
        
    except Exception as e:
        print(f"❌ Automatic fix failed: {e}")
        print("\n📝 Manual fix instructions:")
        print("1. Find this line in your paper_trading_engine.py:")
        print('   f"price ${current_price:.2f} > buy level ${analysis[\'buy_level\']:.2f}"')
        print("2. Change > to <")
        print("3. Find this section:")
        print("   needed = current_price - analysis['buy_level']")
        print("   improvements.append(f\"price drop ${abs(needed):.2f}\")")
        print("4. Replace with:")
        print("   if current_price < analysis['buy_level']:")
        print("       needed = analysis['buy_level'] - current_price")
        print("       improvements.append(f\"price rise ${needed:.2f}\")")
        print("   else:")
        print("       needed = current_price - analysis['buy_level']")
        print("       improvements.append(f\"price drop ${needed:.2f}\")")

if __name__ == "__main__":
    fix_price_direction_logic()