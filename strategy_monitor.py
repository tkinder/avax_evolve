# strategy_monitor.py
import pandas as pd
import sys
import time
from datetime import datetime
import os
from hybrid_data_pipeline import HybridDataPipeline

# Add src to path
sys.path.append('src')

from core.regime_detector import EnhancedRegimeDetector, RegimeBasedTradingRules
from core.adaptive_backtester import AdaptiveParams, AdaptivePriceLevels

class AVAXStrategyMonitor:
    """
    Monitor AVAX for entry signals using your optimized strategy
    """
    
    def __init__(self):
        # Your optimized parameters
        self.params = AdaptiveParams(
            buy_threshold_pct=0.162,
            sell_threshold_pct=0.691,
            regime_short_window=10,
            regime_medium_window=30,
            regime_long_window=60,
            max_position_pct=0.80,
            stop_loss_pct=0.082,
            take_profit_pct=0.150,
        )
        
        # Initialize components
        self.regime_detector = EnhancedRegimeDetector(
            short_window=self.params.regime_short_window,
            medium_window=self.params.regime_medium_window,
            long_window=self.params.regime_long_window
        )
        
        self.price_calculator = AdaptivePriceLevels(self.params)
        self.data_pipeline = HybridDataPipeline()
        
        print("👁️  AVAX Strategy Monitor Initialized")
        print("🎯 Watching for buy signals...")
    
    def get_current_analysis(self):
        """
        Get current market analysis
        """
        try:
            # Get fresh data
            df = self.data_pipeline.get_complete_dataset(historical_days=90)
            
            if df is None or len(df) < self.params.regime_long_window:
                return None
            
            current_idx = len(df) - 1
            current_price = df['close'].iloc[-1]
            
            # Regime detection
            regime = self.regime_detector.detect_regime(df, current_idx)
            
            # Price levels
            buy_level, sell_level, support, resistance = self.price_calculator.get_price_levels(df, current_idx)
            
            # Entry signal analysis
            basic_entry = RegimeBasedTradingRules.should_enter_long(regime, current_price, buy_level)
            confidence_ok = regime.confidence >= 0.6
            
            # Price position
            price_range = max(resistance - support, current_price * 0.01)
            price_position = max(0, min(1, (current_price - support) / price_range))
            
            # Momentum filter
            momentum_filter = True
            if regime.momentum.value == "accel_down":
                momentum_filter = (price_position < 0.25 and regime.trend.value != "strong_bear")
            elif regime.momentum.value == "steady_down":
                momentum_filter = price_position < 0.4
            
            # Trend filter
            trend_filter = True
            if regime.trend.value in ["mild_bear", "strong_bear"]:
                trend_filter = price_position < 0.2
            
            should_enter = (basic_entry and confidence_ok and momentum_filter and trend_filter)
            
            return {
                'timestamp': datetime.now(),
                'price': current_price,
                'regime': regime,
                'buy_level': buy_level,
                'sell_level': sell_level,
                'support': support,
                'resistance': resistance,
                'price_position': price_position,
                'should_enter': should_enter,
                'confidence_ok': confidence_ok,
                'momentum_filter': momentum_filter,
                'trend_filter': trend_filter,
                'basic_entry': basic_entry
            }
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None
    
    def print_status(self, analysis):
        """
        Print current status
        """
        if analysis is None:
            print("❌ Unable to get current analysis")
            return
        
        regime = analysis['regime']
        
        print(f"\n📊 AVAX STRATEGY STATUS - {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"💰 Price: ${analysis['price']:.2f}")
        print(f"🧠 Regime: {regime.trend.value}/{regime.volatility.value}/{regime.momentum.value}")
        print(f"🎯 Confidence: {regime.confidence:.3f}")
        print(f"📈 Buy Level: ${analysis['buy_level']:.2f}")
        print(f"📉 Sell Level: ${analysis['sell_level']:.2f}")
        print(f"📍 Position in Range: {analysis['price_position']:.1%}")
        
        # Signal analysis
        print(f"\n🔍 SIGNAL ANALYSIS:")
        print(f"   Basic Entry: {analysis['basic_entry']}")
        print(f"   Confidence (≥0.6): {analysis['confidence_ok']}")
        print(f"   Momentum Filter: {analysis['momentum_filter']}")
        print(f"   Trend Filter: {analysis['trend_filter']}")
        print(f"   Should Avoid Trading: {regime.should_avoid_trading()}")
        
        # Final signal
        if analysis['should_enter']:
            print(f"🟢 SIGNAL: BUY SIGNAL ACTIVE!")
            
            # Position sizing
            base_position = 10000 * self.params.max_position_pct
            regime_multiplier = RegimeBasedTradingRules.get_position_size_multiplier(regime)
            adjusted_position = base_position * regime_multiplier
            avax_quantity = adjusted_position / analysis['price']
            
            print(f"💰 POSITION SIZING:")
            print(f"   Testnet Balance: $10,000")
            print(f"   Position Size: ${adjusted_position:.2f}")
            print(f"   AVAX Quantity: {avax_quantity:.4f}")
            print(f"   Target Exit: ${analysis['sell_level']:.2f}")
            
            potential_gain = ((analysis['sell_level'] - analysis['price']) / analysis['price']) * 100
            print(f"   Potential Gain: {potential_gain:.1f}%")
            
        else:
            print(f"🔴 SIGNAL: NO BUY (waiting for better conditions)")
            
            # Show what needs to improve
            improvements_needed = []
            if not analysis['confidence_ok']:
                improvements_needed.append(f"Confidence: {regime.confidence:.3f} → 0.60+")
            if not analysis['momentum_filter']:
                improvements_needed.append("Momentum: needs stabilization")
            if not analysis['trend_filter']:
                improvements_needed.append("Trend: too bearish")
            if regime.should_avoid_trading():
                improvements_needed.append("Risk: high-risk conditions")
            
            if improvements_needed:
                print(f"   Needs: {', '.join(improvements_needed)}")
    
    def monitor_once(self):
        """
        Run one monitoring cycle
        """
        analysis = self.get_current_analysis()
        self.print_status(analysis)
        return analysis
    
    def monitor_continuous(self, interval_minutes=60):
        """
        Monitor continuously (for when Pi is running 24/7)
        """
        print(f"🔄 Starting continuous monitoring (checking every {interval_minutes} minutes)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                analysis = self.monitor_once()
                
                # Log to file
                if analysis:
                    log_entry = f"{analysis['timestamp']},{analysis['price']:.2f},{analysis['regime'].trend.value},{analysis['regime'].confidence:.3f},{analysis['should_enter']}\n"
                    
                    with open('avax_monitor_log.csv', 'a') as f:
                        f.write(log_entry)
                
                # Wait for next check
                print(f"\n⏰ Next check in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user")

def main():
    """
    Main monitoring function
    """
    print("👁️  AVAX STRATEGY MONITOR")
    print("=" * 30)
    print("This monitors AVAX for your strategy's buy signals")
    print()
    
    monitor = AVAXStrategyMonitor()
    
    # Run once
    print("🔍 Current analysis:")
    analysis = monitor.monitor_once()
    
    print(f"\n🎯 MONITORING OPTIONS:")
    print("1. One-time check (done above)")
    print("2. Continuous monitoring")
    print("3. Exit")
    
    choice = input("\nChoose option (1/2/3): ").strip()
    
    if choice == "2":
        interval = input("Check interval in minutes (default 60): ").strip()
        interval = int(interval) if interval.isdigit() else 60
        monitor.monitor_continuous(interval)
    else:
        print("🎯 Monitor complete!")

if __name__ == "__main__":
    main()