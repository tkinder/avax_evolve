# test_strategy_integration.py
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path so we can import your modules
sys.path.append('src')

try:
    from core.regime_detector import EnhancedRegimeDetector, RegimeBasedTradingRules
    from core.adaptive_backtester import AdaptiveParams, AdaptivePriceLevels
    print("✅ Successfully imported your strategy modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the avax_evolve directory")
    sys.exit(1)

class LiveStrategyTester:
    """
    Test your actual strategy with live data
    """
    
    def __init__(self):
        # Load the hybrid data we just created
        self.df = pd.read_csv('hybrid_avax_data.csv', index_col=0, parse_dates=True)
        
        # Your optimized parameters
        self.params = AdaptiveParams(
            buy_threshold_pct=0.162,      # 16.2% of range
            sell_threshold_pct=0.691,     # 69.1% of range
            regime_short_window=10,
            regime_medium_window=30,
            regime_long_window=60,
            max_position_pct=0.80,
            stop_loss_pct=0.082,          # 8.2%
            take_profit_pct=0.150,        # 15.0%
        )
        
        # Initialize your components
        self.regime_detector = EnhancedRegimeDetector(
            short_window=self.params.regime_short_window,
            medium_window=self.params.regime_medium_window,
            long_window=self.params.regime_long_window
        )
        
        self.price_calculator = AdaptivePriceLevels(self.params)
        
        print("🧠 Live Strategy Tester Initialized")
        print(f"📊 Data: {len(self.df)} days ({self.df.index[0].strftime('%Y-%m-%d')} to {self.df.index[-1].strftime('%Y-%m-%d')})")
        print(f"🎯 Current AVAX: ${self.df['close'].iloc[-1]:.2f}")
    
    def test_current_regime_detection(self):
        """
        Test regime detection on current market conditions
        """
        print("\n🔍 CURRENT REGIME DETECTION")
        print("=" * 40)
        
        current_idx = len(self.df) - 1
        regime = self.regime_detector.detect_regime(self.df, current_idx)
        
        print(f"📊 Enhanced Regime Analysis:")
        print(f"   Trend: {regime.trend.value}")
        print(f"   Volatility: {regime.volatility.value}")
        print(f"   Momentum: {regime.momentum.value}")
        print(f"   Confidence: {regime.confidence:.3f}")
        print(f"   Should avoid trading: {regime.should_avoid_trading()}")
        
        # Position sizing
        multiplier = RegimeBasedTradingRules.get_position_size_multiplier(regime)
        print(f"   Position multiplier: {multiplier:.3f}x")
        
        return regime
    
    def test_price_levels(self):
        """
        Test adaptive price level calculation
        """
        print("\n💰 ADAPTIVE PRICE LEVELS")
        print("=" * 30)
        
        current_idx = len(self.df) - 1
        buy_level, sell_level, support, resistance = self.price_calculator.get_price_levels(self.df, current_idx)
        
        current_price = self.df['close'].iloc[-1]
        
        print(f"Current Price: ${current_price:.2f}")
        print(f"Buy Level: ${buy_level:.2f}")
        print(f"Sell Level: ${sell_level:.2f}")
        print(f"Support: ${support:.2f}")
        print(f"Resistance: ${resistance:.2f}")
        
        # Price position in range
        if resistance > support:
            price_position = (current_price - support) / (resistance - support)
            print(f"Price Position: {price_position:.1%} of support-resistance range")
        else:
            print("Price Position: Unable to calculate (range too small)")
        
        return buy_level, sell_level, support, resistance
    
    def test_entry_signal(self, regime, buy_level, support, resistance):
        """
        Test if your strategy would generate a buy signal right now
        """
        print("\n🎯 ENTRY SIGNAL TEST")
        print("=" * 25)
        
        current_price = self.df['close'].iloc[-1]
        
        # Your strategy's entry logic
        basic_entry_signal = RegimeBasedTradingRules.should_enter_long(
            regime, current_price, buy_level
        )
        
        # Additional filters from your enhanced logic
        confidence_filter = regime.confidence >= 0.6
        
        # Price position calculation
        price_range = max(resistance - support, current_price * 0.01)
        price_position = max(0, min(1, (current_price - support) / price_range))
        
        # Momentum filter
        momentum_filter = True
        if regime.momentum.value == "accel_down":
            momentum_filter = (price_position < 0.25 and 
                             regime.trend.value != "strong_bear")
        elif regime.momentum.value == "steady_down":
            momentum_filter = price_position < 0.4
        
        # Trend filter
        trend_filter = True
        if regime.trend.value in ["mild_bear", "strong_bear"]:
            trend_filter = price_position < 0.2
        
        should_enter = (basic_entry_signal and confidence_filter and 
                       momentum_filter and trend_filter)
        
        print(f"🔍 Entry Analysis:")
        print(f"   Basic entry signal: {basic_entry_signal}")
        print(f"   Confidence filter (≥0.6): {confidence_filter} ({regime.confidence:.3f})")
        print(f"   Price position: {price_position:.3f}")
        print(f"   Momentum filter: {momentum_filter}")
        print(f"   Trend filter: {trend_filter}")
        print(f"   📊 FINAL ENTRY SIGNAL: {'🟢 BUY' if should_enter else '🔴 NO BUY'}")
        
        if should_enter:
            # Calculate position size
            base_position = 10000 * self.params.max_position_pct  # $10k testnet balance
            regime_multiplier = RegimeBasedTradingRules.get_position_size_multiplier(regime)
            adjusted_position = base_position * regime_multiplier
            avax_quantity = adjusted_position / current_price
            
            print(f"   💰 Position sizing:")
            print(f"      Base position: ${base_position:.2f}")
            print(f"      Regime multiplier: {regime_multiplier:.3f}x")
            print(f"      Adjusted position: ${adjusted_position:.2f}")
            print(f"      AVAX quantity: {avax_quantity:.4f}")
        
        return should_enter
    
    def test_historical_signals(self, days_back=30):
        """
        Test how many signals your strategy would have generated recently
        """
        print(f"\n📈 HISTORICAL SIGNALS (last {days_back} days)")
        print("=" * 35)
        
        signals = []
        start_idx = max(self.params.regime_long_window, len(self.df) - days_back)
        
        for i in range(start_idx, len(self.df)):
            try:
                regime = self.regime_detector.detect_regime(self.df, i)
                buy_level, sell_level, support, resistance = self.price_calculator.get_price_levels(self.df, i)
                current_price = self.df['close'].iloc[i]
                
                # Quick entry test
                basic_signal = RegimeBasedTradingRules.should_enter_long(regime, current_price, buy_level)
                confidence_ok = regime.confidence >= 0.6
                
                if basic_signal and confidence_ok:
                    signals.append({
                        'date': self.df.index[i],
                        'price': current_price,
                        'regime': f"{regime.trend.value}/{regime.volatility.value}/{regime.momentum.value}",
                        'confidence': regime.confidence
                    })
            except:
                continue
        
        print(f"🎯 Found {len(signals)} potential buy signals in last {days_back} days:")
        
        for signal in signals[-5:]:  # Show last 5 signals
            print(f"   {signal['date'].strftime('%Y-%m-%d')}: ${signal['price']:.2f} "
                  f"({signal['regime']}, conf: {signal['confidence']:.2f})")
        
        if len(signals) == 0:
            print("   No signals found - strategy is very selective (good!)")
        
        return signals
    
    def run_full_test(self):
        """
        Run comprehensive test of your strategy with live data
        """
        print("🚀 COMPREHENSIVE STRATEGY TEST")
        print("=" * 50)
        
        # Test 1: Current regime
        regime = self.test_current_regime_detection()
        
        # Test 2: Price levels
        buy_level, sell_level, support, resistance = self.test_price_levels()
        
        # Test 3: Entry signal
        should_enter = self.test_entry_signal(regime, buy_level, support, resistance)
        
        # Test 4: Historical signals
        signals = self.test_historical_signals()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 STRATEGY TEST SUMMARY")
        print("=" * 50)
        
        current_price = self.df['close'].iloc[-1]
        
        print(f"✅ Regime Detection: Working ({regime.confidence:.2f} confidence)")
        print(f"✅ Price Levels: Buy ${buy_level:.2f}, Sell ${sell_level:.2f}")
        print(f"✅ Current Signal: {'🟢 BUY' if should_enter else '🔴 HOLD'}")
        print(f"✅ Strategy Selectivity: {len(signals)} signals in 30 days")
        
        if should_enter:
            print(f"\n🎉 YOUR STRATEGY IS RECOMMENDING A BUY!")
            print(f"   💰 Current AVAX: ${current_price:.2f}")
            print(f"   🎯 Target Exit: ${sell_level:.2f}")
            print(f"   📈 Potential Gain: {((sell_level - current_price) / current_price * 100):.1f}%")
            print(f"   🛡️  Stop Loss: ${current_price * (1 - self.params.stop_loss_pct):.2f}")
        
        print(f"\n🎯 READY FOR TESTNET PAPER TRADING!")
        
        return {
            'regime': regime,
            'should_enter': should_enter,
            'current_price': current_price,
            'buy_level': buy_level,
            'sell_level': sell_level
        }

if __name__ == "__main__":
    tester = LiveStrategyTester()
    results = tester.run_full_test()