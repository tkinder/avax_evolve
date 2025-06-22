# laptop_monitor.py - Optimized for laptop use
import pandas as pd
import sys
import time
import os
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append('src')

from hybrid_data_pipeline import HybridDataPipeline
from core.regime_detector import EnhancedRegimeDetector, RegimeBasedTradingRules
from core.adaptive_backtester import AdaptiveParams, AdaptivePriceLevels

class LaptopAVAXMonitor:
    """
    AVAX monitoring optimized for laptop usage
    """
    
    def __init__(self):
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
        
        self.regime_detector = EnhancedRegimeDetector(
            short_window=self.params.regime_short_window,
            medium_window=self.params.regime_medium_window,
            long_window=self.params.regime_long_window
        )
        
        self.price_calculator = AdaptivePriceLevels(self.params)
        self.data_pipeline = HybridDataPipeline()
        
        # History tracking
        self.history_file = 'avax_monitor_history.json'
        self.load_history()
        
        print("💻 Laptop AVAX Monitor Ready")
    
    def load_history(self):
        """Load previous monitoring history"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            else:
                self.history = []
        except:
            self.history = []
    
    def save_history(self, analysis):
        """Save analysis to history"""
        try:
            record = {
                'timestamp': analysis['timestamp'].isoformat(),
                'price': analysis['price'],
                'trend': analysis['regime'].trend.value,
                'volatility': analysis['regime'].volatility.value,
                'momentum': analysis['regime'].momentum.value,
                'confidence': analysis['regime'].confidence,
                'should_enter': analysis['should_enter'],
                'buy_level': analysis['buy_level'],
                'sell_level': analysis['sell_level']
            }
            
            self.history.append(record)
            
            # Keep only last 100 records
            if len(self.history) > 100:
                self.history = self.history[-100:]
            
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Could not save history: {e}")
    
    def quick_check(self):
        """Quick status check - optimized for frequent use"""
        try:
            print("🔄 Getting latest data...")
            df = self.data_pipeline.get_complete_dataset(historical_days=90)
            
            if df is None:
                print("❌ Could not get data")
                return None
            
            current_idx = len(df) - 1
            current_price = df['close'].iloc[-1]
            regime = self.regime_detector.detect_regime(df, current_idx)
            buy_level, sell_level, support, resistance = self.price_calculator.get_price_levels(df, current_idx)
            
            # Quick entry analysis
            basic_entry = RegimeBasedTradingRules.should_enter_long(regime, current_price, buy_level)
            confidence_ok = regime.confidence >= 0.6
            should_enter = basic_entry and confidence_ok and not regime.should_avoid_trading()
            
            analysis = {
                'timestamp': datetime.now(),
                'price': current_price,
                'regime': regime,
                'buy_level': buy_level,
                'sell_level': sell_level,
                'should_enter': should_enter,
                'confidence_ok': confidence_ok
            }
            
            # Save to history
            self.save_history(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Quick check failed: {e}")
            return None
    
    def print_quick_status(self, analysis):
        """Print concise status"""
        if not analysis:
            return
        
        regime = analysis['regime']
        
        print(f"\n📊 AVAX Quick Check - {analysis['timestamp'].strftime('%H:%M:%S')}")
        print("=" * 45)
        print(f"💰 Price: ${analysis['price']:.2f}")
        print(f"🧠 Regime: {regime.trend.value}/{regime.volatility.value}/{regime.momentum.value}")
        print(f"🎯 Confidence: {regime.confidence:.3f} {'✅' if analysis['confidence_ok'] else '❌'}")
        print(f"📈 Buy Level: ${analysis['buy_level']:.2f}")
        
        if analysis['should_enter']:
            print("🟢 STATUS: BUY SIGNAL ACTIVE! 🚨")
            potential_gain = ((analysis['sell_level'] - analysis['price']) / analysis['price']) * 100
            print(f"   Target: ${analysis['sell_level']:.2f} ({potential_gain:.1f}% gain)")
        else:
            print("🔴 STATUS: No signal (waiting)")
            
            # Show what's missing
            if not analysis['confidence_ok']:
                print(f"   Need confidence: {regime.confidence:.3f} → 0.60")
            if regime.should_avoid_trading():
                print(f"   Risk level: Too high")
    
    def show_recent_history(self, days=7):
        """Show recent monitoring history"""
        if not self.history:
            print("📝 No history available")
            return
        
        cutoff = datetime.now() - timedelta(days=days)
        recent = [h for h in self.history if datetime.fromisoformat(h['timestamp']) > cutoff]
        
        if not recent:
            print(f"📝 No history in last {days} days")
            return
        
        print(f"\n📈 RECENT HISTORY (last {days} days):")
        print("Date       Time   Price   Confidence  Signal")
        print("-" * 45)
        
        for record in recent[-10:]:  # Last 10 records
            dt = datetime.fromisoformat(record['timestamp'])
            signal = "🟢 BUY" if record['should_enter'] else "🔴 --"
            print(f"{dt.strftime('%m-%d')} {dt.strftime('%H:%M')} ${record['price']:6.2f}  {record['confidence']:8.3f}  {signal}")
    
    def set_alert_conditions(self):
        """Set conditions for alerts"""
        print("\n🔔 ALERT SETUP")
        print("Set conditions for when you want to be notified:")
        
        try:
            price_threshold = input("Alert if price drops below $ (or Enter to skip): ").strip()
            confidence_threshold = input("Alert if confidence rises above (0.6): ").strip() or "0.6"
            
            alerts = {
                'price_below': float(price_threshold) if price_threshold else None,
                'confidence_above': float(confidence_threshold)
            }
            
            with open('alert_settings.json', 'w') as f:
                json.dump(alerts, f)
            
            print("✅ Alert settings saved")
            
        except Exception as e:
            print(f"❌ Alert setup failed: {e}")
    
    def check_alerts(self, analysis):
        """Check if any alerts should trigger"""
        try:
            if not os.path.exists('alert_settings.json'):
                return
            
            with open('alert_settings.json', 'r') as f:
                alerts = json.load(f)
            
            triggered = []
            
            if alerts.get('price_below') and analysis['price'] <= alerts['price_below']:
                triggered.append(f"💰 Price alert: ${analysis['price']:.2f} ≤ ${alerts['price_below']:.2f}")
            
            if alerts.get('confidence_above') and analysis['regime'].confidence >= alerts['confidence_above']:
                triggered.append(f"🎯 Confidence alert: {analysis['regime'].confidence:.3f} ≥ {alerts['confidence_above']:.3f}")
            
            if analysis['should_enter']:
                triggered.append("🚨 BUY SIGNAL TRIGGERED!")
            
            if triggered:
                print("\n🔔 ALERTS:")
                for alert in triggered:
                    print(f"   {alert}")
                
                # You could add email/SMS notifications here
                
        except Exception as e:
            print(f"⚠️ Alert check failed: {e}")

def main():
    """Main laptop monitoring interface"""
    print("💻 AVAX Laptop Monitor")
    print("=" * 25)
    
    monitor = LaptopAVAXMonitor()
    
    while True:
        print("\n🎯 MONITORING OPTIONS:")
        print("1. Quick check (current status)")
        print("2. Detailed analysis")
        print("3. Recent history")
        print("4. Set alerts")
        print("5. Continuous monitoring")
        print("6. Exit")
        
        choice = input("\nChoose option (1-6): ").strip()
        
        if choice == "1":
            analysis = monitor.quick_check()
            monitor.print_quick_status(analysis)
            if analysis:
                monitor.check_alerts(analysis)
        
        elif choice == "2":
            # Run full analysis
            os.system("python test_strategy_integration.py")
        
        elif choice == "3":
            days = input("Show history for how many days (7): ").strip()
            days = int(days) if days.isdigit() else 7
            monitor.show_recent_history(days)
        
        elif choice == "4":
            monitor.set_alert_conditions()
        
        elif choice == "5":
            print("🔄 Starting continuous monitoring...")
            print("Press Ctrl+C to stop")
            try:
                while True:
                    analysis = monitor.quick_check()
                    monitor.print_quick_status(analysis)
                    if analysis:
                        monitor.check_alerts(analysis)
                    
                    print("\n⏰ Waiting 30 minutes... (Ctrl+C to stop)")
                    time.sleep(30 * 60)  # 30 minutes
                    
            except KeyboardInterrupt:
                print("\n🛑 Continuous monitoring stopped")
        
        elif choice == "6":
            print("👋 Monitor closed")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()