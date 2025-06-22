# scheduled_monitor.py - Run this on a schedule
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

from laptop_monitor_with_alerts import LaptopAVAXMonitorWithAlerts

def scheduled_check():
    """
    Single check designed to be run by scheduler
    """
    print(f"🕐 Scheduled AVAX Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        monitor = LaptopAVAXMonitorWithAlerts()
        
        # Quick check with alerts
        analysis = monitor.quick_check(send_alerts=True)
        
        if analysis:
            regime = analysis['regime']
            
            # Concise log output
            print(f"💰 Price: ${analysis['price']:.2f}")
            print(f"🧠 Regime: {regime.trend.value}/{regime.volatility.value}/{regime.momentum.value}")
            print(f"🎯 Confidence: {regime.confidence:.3f}")
            print(f"📊 Signal: {'🟢 BUY' if analysis['should_enter'] else '🔴 No signal'}")
            
            # Log to file for history
            log_entry = f"{datetime.now().isoformat()},{analysis['price']:.2f},{regime.trend.value},{regime.confidence:.3f},{analysis['should_enter']}\n"
            
            with open('scheduled_monitor_log.csv', 'a') as f:
                # Add header if file is new
                if os.path.getsize('scheduled_monitor_log.csv') == 0:
                    f.write("timestamp,price,trend,confidence,should_enter\n")
                f.write(log_entry)
            
            if analysis['should_enter']:
                print("🚨 BUY SIGNAL DETECTED! Check your alerts!")
            
            print("✅ Check complete")
        else:
            print("❌ Could not get market data")
            
    except Exception as e:
        print(f"❌ Scheduled check failed: {e}")
        
        # Log error
        with open('scheduled_monitor_errors.log', 'a') as f:
            f.write(f"{datetime.now().isoformat()}: {e}\n")

if __name__ == "__main__":
    scheduled_check()