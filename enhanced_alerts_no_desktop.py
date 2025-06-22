# enhanced_alerts_no_desktop.py
import smtplib
import json
import os
import time
import sys
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EnhancedAlertSystem:
    """
    Alert system optimized for headless/SSH environments
    """
    
    def __init__(self):
        self.alert_config_file = 'alert_config.json'
        self.alert_history_file = 'alert_history.json'
        self.config = None  # Initialize as None
        self.history = []   # Initialize as empty list
        self.load_config()
        self.load_history()
    
    def load_config(self):
        """Load alert configuration"""
        try:
            if os.path.exists(self.alert_config_file):
                with open(self.alert_config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Ensure all required keys exist
                    self.config = self.create_default_config()
                    self.config.update(loaded_config)
            else:
                self.config = self.create_default_config()
                self.save_config()
        except Exception as e:
            print(f"Warning: Could not load config ({e}), using defaults")
            self.config = self.create_default_config()
            try:
                self.save_config()
            except:
                pass
    
    def create_default_config(self):
        """Create default alert configuration"""
        return {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': '',
                'sender_password': '',
                'recipient_email': ''
            },
            'console': {
                'enabled': True,
                'clear_screen': True,
                'sound_alerts': True,
                'require_acknowledgment': True
            },
            'file': {
                'enabled': True,
                'alert_file': 'AVAX_ALERTS.txt'
            },
            'thresholds': {
                'confidence_above': 0.6,
                'price_below': 16.50,
                'price_above': 25.00,
                'buy_signal': True,
                'regime_change': True
            },
            'cooldown_minutes': 60
        }
    
    def save_config(self):
        """Save alert configuration"""
        try:
            with open(self.alert_config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Config saved to {self.alert_config_file}")
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
    
    def load_history(self):
        """Load alert history"""
        try:
            if os.path.exists(self.alert_history_file):
                with open(self.alert_history_file, 'r') as f:
                    self.history = json.load(f)
            else:
                self.history = []
        except Exception as e:
            print(f"Warning: Could not load history ({e}), starting fresh")
            self.history = []
    
    def save_history(self, alert_type, message):
        """Save alert to history"""
        try:
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'type': alert_type,
                'message': message
            })
            
            if len(self.history) > 100:
                self.history = self.history[-100:]
            
            with open(self.alert_history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
    
    def should_send_alert(self, alert_type):
        """Check cooldown logic"""
        try:
            if not self.history:
                return True
            
            cooldown_minutes = self.config.get('cooldown_minutes', 60)
            now = datetime.now()
            
            for alert in reversed(self.history[-10:]):
                alert_time = datetime.fromisoformat(alert['timestamp'])
                minutes_ago = (now - alert_time).total_seconds() / 60
                
                if alert['type'] == alert_type and minutes_ago < cooldown_minutes:
                    return False
            
            return True
        except Exception as e:
            print(f"Warning: Cooldown check failed ({e}), allowing alert")
            return True
    
    def send_console_alert(self, title, message, alert_type="info"):
        """Send enhanced console alert"""
        try:
            # Safely get console config
            console_config = self.config.get('console', {})
            if not console_config.get('enabled', True):
                return False
            
            # Clear screen for critical alerts
            if alert_type == "buy_signal" and console_config.get('clear_screen', True):
                os.system('clear' if os.name == 'posix' else 'cls')
            
            # Color codes for visual impact
            colors = {
                "buy_signal": "\033[1;92m",  # Bold bright green
                "warning": "\033[1;93m",     # Bold yellow
                "info": "\033[1;94m",        # Bold blue
                "error": "\033[1;91m",       # Bold red
                "test": "\033[1;96m",        # Bold cyan
                "reset": "\033[0m"
            }
            
            color = colors.get(alert_type, colors["info"])
            reset = colors["reset"]
            
            # Create visual alert
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n{color}")
            print("🚨" * 30)
            print(f"⚠️  AVAX TRADING ALERT: {title}")
            print(f"📅 {timestamp}")
            print("═" * 60)
            print(message)
            print("═" * 60)
            
            if alert_type == "buy_signal":
                print("🚨 🚨 BUY SIGNAL DETECTED! 🚨 🚨")
                print("💰 CHECK YOUR TRADING PLATFORM IMMEDIATELY!")
                print("📧 Check your email for details!")
                
                # Sound alert
                if console_config.get('sound_alerts', True):
                    self.make_sound_alert()
                
                # Require acknowledgment
                if console_config.get('require_acknowledgment', True):
                    print("\n🔔 Press Enter to acknowledge this alert...")
                    input()
            
            print("🚨" * 30)
            print(f"{reset}\n")
            
            return True
            
        except Exception as e:
            print(f"Console alert failed: {e}")
            return False
    
    def make_sound_alert(self):
        """Make audible alert"""
        try:
            # Multiple beeps for attention
            for i in range(5):
                print("\a", end="", flush=True)
                time.sleep(0.3)
        except:
            pass
    
    def send_file_alert(self, title, message):
        """Write alert to file for persistent notification"""
        try:
            file_config = self.config.get('file', {})
            if not file_config.get('enabled', True):
                return False
            
            alert_file = file_config.get('alert_file', 'AVAX_ALERTS.txt')
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            alert_content = f"""
╔══════════════════════════════════════════════════════════════╗
║                        AVAX TRADING ALERT                    ║
╠══════════════════════════════════════════════════════════════╣
║ Alert: {title:<50} ║
║ Time:  {timestamp:<50} ║
╠══════════════════════════════════════════════════════════════╣
║ {message:<60} ║
╚══════════════════════════════════════════════════════════════╝

"""
            
            # Append to alert file
            with open(alert_file, 'a') as f:
                f.write(alert_content)
            
            # Also create a simple status file
            with open('LATEST_AVAX_ALERT.txt', 'w') as f:
                f.write(f"{timestamp}: {title}\n{message}")
            
            return True
            
        except Exception as e:
            print(f"File alert failed: {e}")
            return False
    
    def send_email_alert(self, subject, message):
        """Send email alert"""
        try:
            email_config = self.config.get('email', {})
            if not email_config.get('enabled', False):
                return False
            
            sender = email_config.get('sender_email', '')
            recipient = email_config.get('recipient_email', '')
            password = email_config.get('sender_password', '')
            
            if not all([sender, recipient, password]):
                print("   ⚠️  Email not configured (missing credentials)")
                return False
            
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(email_config.get('smtp_server', 'smtp.gmail.com'), 
                                email_config.get('smtp_port', 587))
            server.starttls()
            server.login(sender, password)
            text = msg.as_string()
            server.sendmail(sender, recipient, text)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Email alert failed: {e}")
            return False
    
    def send_alert(self, alert_type, title, message):
        """Send alert through all enabled channels"""
        try:
            if not self.should_send_alert(alert_type):
                print(f"⏰ Skipping {alert_type} alert (cooldown)")
                return
            
            sent_count = 0
            
            # Console alert (always try this)
            if self.send_console_alert(title, message, alert_type):
                sent_count += 1
                print("   ✅ Console alert displayed")
            
            # Email alert
            email_config = self.config.get('email', {})
            if email_config.get('enabled', False):
                if self.send_email_alert(f"AVAX Strategy: {title}", message):
                    sent_count += 1
                    print("   ✅ Email sent")
                else:
                    print("   ❌ Email failed or not configured")
            else:
                print("   ℹ️  Email disabled")
            
            # File alert
            if self.send_file_alert(title, message):
                sent_count += 1
                print("   ✅ File alert created")
            
            # Save to history
            self.save_history(alert_type, f"{title}: {message}")
            
            print(f"   📊 Alert sent via {sent_count} method(s)")
            
            # Show file locations for reference
            if alert_type in ["buy_signal", "test"]:
                file_config = self.config.get('file', {})
                print(f"\n📁 Alert files created:")
                print(f"   📄 {file_config.get('alert_file', 'AVAX_ALERTS.txt')}")
                print(f"   📄 LATEST_AVAX_ALERT.txt")
                email_config = self.config.get('email', {})
                if email_config.get('enabled', False):
                    print(f"   📧 Email sent to {email_config.get('recipient_email', 'N/A')}")
        
        except Exception as e:
            print(f"Send alert failed: {e}")
    
    def check_alerts(self, analysis):
        """Check if any alerts should be triggered"""
        try:
            if not analysis:
                return
            
            regime = analysis['regime']
            price = analysis['price']
            confidence = regime.confidence
            thresholds = self.config.get('thresholds', {})
            
            # Buy signal alert (highest priority)
            if analysis['should_enter'] and thresholds.get('buy_signal', True):
                message = f"""
🚨 BUY SIGNAL ACTIVE! 🚨

💰 Current Price: ${price:.2f}
📈 Buy Level: ${analysis['buy_level']:.2f}
🎯 Target Price: ${analysis['sell_level']:.2f}
📊 Potential Gain: {((analysis['sell_level'] - price) / price * 100):.1f}%

🧠 Market Regime: {regime.trend.value}/{regime.volatility.value}/{regime.momentum.value}
🎯 Confidence: {confidence:.3f}

⚡ ACTION REQUIRED:
1. Check your Binance.US testnet account
2. Execute trade if conditions still valid
3. Monitor position according to strategy rules

💡 This alert will be in your email and AVAX_ALERTS.txt file
                """
                self.send_alert('buy_signal', 'AVAX BUY SIGNAL', message)
            
            # Confidence improvement alert
            elif confidence >= thresholds.get('confidence_above', 0.6):
                message = f"Confidence reached {confidence:.3f}\nPrice: ${price:.2f}\nRegime: {regime.trend.value}/{regime.volatility.value}"
                self.send_alert('confidence', 'High Confidence Detected', message)
            
            # Price alerts
            elif price <= thresholds.get('price_below', 0):
                message = f"Price dropped to ${price:.2f}\nBelow threshold: ${thresholds['price_below']:.2f}"
                self.send_alert('price_low', 'Price Alert - Low', message)
        
        except Exception as e:
            print(f"Check alerts failed: {e}")

def test_enhanced_alerts():
    """Test the enhanced alert system"""
    print("🧪 Testing Enhanced Alert System (No Desktop Required)")
    print("=" * 60)
    
    try:
        alert_system = EnhancedAlertSystem()
        
        # Test console + file + email alerts
        alert_system.send_alert(
            'test', 
            'System Test Alert',
            'Testing enhanced alert system!\n\nThis should appear in:\n✅ Console (this screen)\n✅ Email (if configured)\n✅ File (AVAX_ALERTS.txt)'
        )
        
        print("\n📁 Check these files for alerts:")
        print("   📄 AVAX_ALERTS.txt")
        print("   📄 LATEST_AVAX_ALERT.txt")
        print("   📄 alert_config.json (configuration)")
        print("   📄 alert_history.json (alert history)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_alerts()