# laptop_monitor_with_alerts.py
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
from advanced_alerts import AlertSystem

class LaptopAVAXMonitorWithAlerts:
    """
    AVAX monitoring with integrated alert system
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
        
        # Initialize alert system
        self.alert_system = AlertSystem()
        
        # History tracking
        self.history_file = 'avax_monitor_history.json'
        self.load_history()
        
        print("💻 Laptop AVAX Monitor with Alerts Ready")
    
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
    
    def quick_check(self, send_alerts=True):
        """Quick status check with optional alerts"""
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
            
            # Enhanced entry analysis
            basic_entry = RegimeBasedTradingRules.should_enter_long(regime, current_price, buy_level)
            confidence_ok = regime.confidence >= 0.6
            
            # Additional filters
            price_range = max(resistance - support, current_price * 0.01)
            price_position = max(0, min(1, (current_price - support) / price_range))
            
            momentum_filter = True
            if regime.momentum.value == "accel_down":
                momentum_filter = (price_position < 0.25 and regime.trend.value != "strong_bear")
            elif regime.momentum.value == "steady_down":
                momentum_filter = price_position < 0.4
            
            trend_filter = True
            if regime.trend.value in ["mild_bear", "strong_bear"]:
                trend_filter = price_position < 0.2
            
            should_enter = (basic_entry and confidence_ok and momentum_filter and 
                          trend_filter and not regime.should_avoid_trading())
            
            analysis = {
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
            
            # Save to history
            self.save_history(analysis)
            
            # Check for alerts
            if send_alerts:
                self.alert_system.check_alerts(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Quick check failed: {e}")
            return None
    
    def print_detailed_status(self, analysis):
        """Print detailed status with alert info"""
        if not analysis:
            return
        
        regime = analysis['regime']
        
        print(f"\n📊 DETAILED AVAX STATUS - {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 65)
        print(f"💰 Current Price: ${analysis['price']:.2f}")
        print(f"🧠 Regime: {regime.trend.value}/{regime.volatility.value}/{regime.momentum.value}")
        print(f"🎯 Confidence: {regime.confidence:.3f} {'✅' if analysis['confidence_ok'] else '❌'}")
        print(f"📈 Buy Level: ${analysis['buy_level']:.2f}")
        print(f"📉 Sell Level: ${analysis['sell_level']:.2f}")
        print(f"🏠 Support: ${analysis['support']:.2f}")
        print(f"🏔️  Resistance: ${analysis['resistance']:.2f}")
        print(f"📍 Position in Range: {analysis['price_position']:.1%}")
        
        # Detailed signal analysis
        print(f"\n🔍 SIGNAL BREAKDOWN:")
        print(f"   Basic Entry Signal: {analysis['basic_entry']} {'✅' if analysis['basic_entry'] else '❌'}")
        print(f"   Confidence Filter: {analysis['confidence_ok']} {'✅' if analysis['confidence_ok'] else '❌'}")
        print(f"   Momentum Filter: {analysis['momentum_filter']} {'✅' if analysis['momentum_filter'] else '❌'}")
        print(f"   Trend Filter: {analysis['trend_filter']} {'✅' if analysis['trend_filter'] else '❌'}")
        print(f"   Risk Level: {'✅ Safe' if not regime.should_avoid_trading() else '❌ High Risk'}")
        
        # Final signal
        if analysis['should_enter']:
            print(f"\n🟢 FINAL SIGNAL: BUY SIGNAL ACTIVE! 🚨")
            
            # Position sizing
            base_position = 10000 * self.params.max_position_pct
            regime_multiplier = RegimeBasedTradingRules.get_position_size_multiplier(regime)
            adjusted_position = base_position * regime_multiplier
            avax_quantity = adjusted_position / analysis['price']
            
            print(f"💰 POSITION DETAILS:")
            print(f"   Testnet Balance: $10,000")
            print(f"   Max Position: {self.params.max_position_pct:.1%}")
            print(f"   Regime Multiplier: {regime_multiplier:.3f}x")
            print(f"   Adjusted Position: ${adjusted_position:.2f}")
            print(f"   AVAX Quantity: {avax_quantity:.4f}")
            print(f"   Entry Price: ${analysis['price']:.2f}")
            print(f"   Target Exit: ${analysis['sell_level']:.2f}")
            print(f"   Stop Loss: ${analysis['price'] * (1 - self.params.stop_loss_pct):.2f}")
            
            potential_gain = ((analysis['sell_level'] - analysis['price']) / analysis['price']) * 100
            risk_amount = adjusted_position * self.params.stop_loss_pct
            
            print(f"   Potential Gain: {potential_gain:.1f}%")
            print(f"   Risk Amount: ${risk_amount:.2f}")
            print(f"   Risk/Reward: 1:{(potential_gain / (self.params.stop_loss_pct * 100)):.1f}")
            
        else:
            print(f"\n🔴 FINAL SIGNAL: NO BUY (waiting for optimal conditions)")
            
            # Show specific improvements needed
            improvements = []
            if not analysis['confidence_ok']:
                improvements.append(f"Confidence: {regime.confidence:.3f} → 0.60+")
            if not analysis['momentum_filter']:
                improvements.append("Momentum: stabilization needed")
            if not analysis['trend_filter']:
                improvements.append("Trend: less bearish conditions")
            if regime.should_avoid_trading():
                improvements.append("Risk: reduce market stress")
            
            if improvements:
                print(f"   🎯 Need improvements: {', '.join(improvements)}")
        
        # Alert status
        alert_config = self.alert_system.config
        enabled_alerts = []
        if alert_config['email']['enabled']:
            enabled_alerts.append('📧 Email')
        if alert_config['discord']['enabled']:
            enabled_alerts.append('💬 Discord')
        if alert_config['slack']['enabled']:
            enabled_alerts.append('💼 Slack')
        if alert_config['desktop']['enabled']:
            enabled_alerts.append('🖥️ Desktop')
        
        print(f"\n🔔 Alert Status: {', '.join(enabled_alerts) if enabled_alerts else 'Desktop only'}")
    
    def show_alert_history(self, days=7):
        """Show recent alert history"""
        try:
            if not os.path.exists(self.alert_system.alert_history_file):
                print("📝 No alert history available")
                return
            
            with open(self.alert_system.alert_history_file, 'r') as f:
                alert_history = json.load(f)
            
            if not alert_history:
                print("📝 No alerts sent yet")
                return
            
            cutoff = datetime.now() - timedelta(days=days)
            recent_alerts = [a for a in alert_history 
                           if datetime.fromisoformat(a['timestamp']) > cutoff]
            
            if not recent_alerts:
                print(f"📝 No alerts in last {days} days")
                return
            
            print(f"\n🔔 RECENT ALERTS (last {days} days):")
            print("Date       Time   Type           Message")
            print("-" * 60)
            
            for alert in recent_alerts[-10:]:
                dt = datetime.fromisoformat(alert['timestamp'])
                message_preview = alert['message'][:30] + "..." if len(alert['message']) > 30 else alert['message']
                print(f"{dt.strftime('%m-%d')} {dt.strftime('%H:%M')} {alert['type']:12} {message_preview}")
                
        except Exception as e:
            print(f"❌ Could not load alert history: {e}")
    
    def setup_alerts_interactive(self):
        """Interactive alert setup"""
        from advanced_alerts import setup_alerts
        self.alert_system = setup_alerts()
        print("✅ Alert system updated")
    
    def test_alerts(self):
        """Test alert system"""
        print("🧪 Testing alert system...")
        self.alert_system.send_alert('test', 'AVAX Monitor Test', 
                                    f'Alert system test at {datetime.now().strftime("%H:%M:%S")}')

def main():
    """Enhanced main interface with alerts"""
    print("💻 AVAX Laptop Monitor with Alerts")
    print("=" * 35)
    
    monitor = LaptopAVAXMonitorWithAlerts()
    
    while True:
        print("\n🎯 MONITORING OPTIONS:")
        print("1. Quick check (with alerts)")
        print("2. Detailed analysis")
        print("3. Recent history")
        print("4. Alert history") 
        print("5. Setup/modify alerts")
        print("6. Test alerts")
        print("7. Continuous monitoring")
        print("8. Exit")
        
        choice = input("\nChoose option (1-8): ").strip()
        
        if choice == "1":
            analysis = monitor.quick_check(send_alerts=True)
            if analysis:
                monitor.print_detailed_status(analysis)
        
        elif choice == "2":
            analysis = monitor.quick_check(send_alerts=False)
            if analysis:
                monitor.print_detailed_status(analysis)
        
        elif choice == "3":
            days = input("Show history for how many days (7): ").strip()
            days = int(days) if days.isdigit() else 7
            monitor.show_recent_history(days)
        
        elif choice == "4":
            days = input("Show alert history for how many days (7): ").strip()
            days = int(days) if days.isdigit() else 7
            monitor.show_alert_history(days)
        
        elif choice == "5":
            monitor.setup_alerts_interactive()
        
        elif choice == "6":
            monitor.test_alerts()
        
        elif choice == "7":
            print("🔄 Starting continuous monitoring with alerts...")
            interval = input("Check interval in minutes (30): ").strip()
            interval = int(interval) if interval.isdigit() else 30
            
            print(f"Running continuous monitoring every {interval} minutes")
            print("Press Ctrl+C to stop")
            
            try:
                while True:
                    print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    analysis = monitor.quick_check(send_alerts=True)
                    
                    if analysis:
                        # Print concise status for continuous monitoring
                        regime = analysis['regime']
                        signal_emoji = "🟢" if analysis['should_enter'] else "🔴"
                        
                        print(f"{signal_emoji} ${analysis['price']:.2f} | "
                              f"{regime.trend.value}/{regime.volatility.value} | "
                              f"Conf: {regime.confidence:.3f} | "
                              f"{'BUY SIGNAL' if analysis['should_enter'] else 'No signal'}")
                    
                    print(f"💤 Sleeping {interval} minutes... (Ctrl+C to stop)")
                    time.sleep(interval * 60)
                    
            except KeyboardInterrupt:
                print("\n🛑 Continuous monitoring stopped")
        
        elif choice == "8":
            print("👋 Monitor closed")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()