# advanced_alerts.py
import smtplib
import json
import os
import time
import requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertSystem:
    """
    Multi-channel alert system for AVAX strategy
    Compatible with laptop_monitor_with_alerts.py
    """
    
    def __init__(self):
        self.alert_config_file = 'alert_config.json'
        self.alert_history_file = 'alert_history.json'
        self.config = None
        self.history = []
        self.load_config()
        self.load_history()
    
    def load_config(self):
        """Load alert configuration with error handling"""
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
                'sender_password': '',  # Use app password for Gmail
                'recipient_email': ''
            },
            'discord': {
                'enabled': False,
                'webhook_url': ''
            },
            'slack': {
                'enabled': False,
                'webhook_url': ''
            },
            'desktop': {
                'enabled': True,  # Always available
                'clear_screen': False,  # Don't clear screen in continuous mode
                'sound_alerts': True,
                'require_acknowledgment': False  # Don't block in continuous mode
            },
            'console': {  # Alias for desktop for compatibility
                'enabled': True,
                'clear_screen': False,
                'sound_alerts': True,
                'require_acknowledgment': False
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
            'cooldown_minutes': 60  # 60 minutes for production, 1 for testing
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
        """Load alert history to prevent spam"""
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
            
            # Keep only last 100 alerts
            if len(self.history) > 100:
                self.history = self.history[-100:]
            
            with open(self.alert_history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
    
    def should_send_alert(self, alert_type):
        """Check if we should send alert (cooldown logic)"""
        try:
            if not self.history:
                return True
            
            cooldown_minutes = self.config.get('cooldown_minutes', 60)
            now = datetime.now()
            
            # Check recent alerts of same type
            for alert in reversed(self.history[-10:]):
                alert_time = datetime.fromisoformat(alert['timestamp'])
                minutes_ago = (now - alert_time).total_seconds() / 60
                
                if alert['type'] == alert_type and minutes_ago < cooldown_minutes:
                    return False
            
            return True
        except Exception as e:
            print(f"Warning: Cooldown check failed ({e}), allowing alert")
            return True
    
    def send_desktop_notification(self, title, message):
        """Send desktop notification (cross-platform)"""
        try:
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                os.system(f'''osascript -e 'display notification "{message}" with title "{title}"' ''')
            elif system == "Linux":
                # Try multiple notification methods
                try:
                    os.system(f'notify-send "{title}" "{message}"')
                except:
                    try:
                        os.system(f'zenity --notification --text="{title}: {message}"')
                    except:
                        pass
            elif system == "Windows":
                try:
                    import win10toast
                    toaster = win10toast.ToastNotifier()
                    toaster.show_toast(title, message, duration=10)
                except ImportError:
                    print(f"💡 Install win10toast for Windows notifications: pip install win10toast")
            
            return True
        except Exception as e:
            print(f"Desktop notification failed: {e}")
            return False
    
    def send_console_alert(self, title, message, alert_type="info"):
        """Send enhanced console alert - for compatibility with monitor"""
        try:
            console_config = self.config.get('console', self.config.get('desktop', {}))
            if not console_config.get('enabled', True):
                return False
            
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
                
                # Sound alert
                if console_config.get('sound_alerts', True):
                    self.make_sound_alert()
                
                # Optional acknowledgment (disabled for continuous monitoring)
                if console_config.get('require_acknowledgment', False):
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
            for i in range(3):  # Reduced for continuous monitoring
                print("\a", end="", flush=True)
                time.sleep(0.2)
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
            
            with open(alert_file, 'a') as f:
                f.write(alert_content)
            
            with open('LATEST_AVAX_ALERT.txt', 'w') as f:
                f.write(f"{timestamp}: {title}\n{message}")
            
            return True
            
        except Exception as e:
            print(f"File alert failed: {e}")
            return False
    
    def send_email_alert(self, subject, message):
        """Send email alert"""
        try:
            if not self.config['email']['enabled']:
                return False
            
            sender = self.config['email']['sender_email']
            recipient = self.config['email']['recipient_email']
            password = self.config['email']['sender_password']
            
            if not all([sender, recipient, password]):
                print("❌ Email configuration incomplete")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config['email']['smtp_server'], 
                                self.config['email']['smtp_port'])
            server.starttls()
            server.login(sender, password)
            text = msg.as_string()
            server.sendmail(sender, recipient, text)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Email alert failed: {e}")
            return False
    
    def send_discord_alert(self, message):
        """Send Discord webhook alert"""
        try:
            if not self.config['discord']['enabled']:
                return False
            
            webhook_url = self.config['discord']['webhook_url']
            if not webhook_url:
                return False
            
            data = {
                "content": message,
                "username": "AVAX Strategy Bot"
            }
            
            response = requests.post(webhook_url, json=data)
            return response.status_code == 204
            
        except Exception as e:
            print(f"Discord alert failed: {e}")
            return False
    
    def send_slack_alert(self, message):
        """Send Slack webhook alert"""
        try:
            if not self.config['slack']['enabled']:
                return False
            
            webhook_url = self.config['slack']['webhook_url']
            if not webhook_url:
                return False
            
            data = {
                "text": message,
                "username": "AVAX Strategy Bot",
                "icon_emoji": ":chart_with_upwards_trend:"
            }
            
            response = requests.post(webhook_url, json=data)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Slack alert failed: {e}")
            return False
    
    def send_alert(self, alert_type, title, message):
        """Send alert through all enabled channels"""
        try:
            if not self.should_send_alert(alert_type):
                print(f"⏰ Skipping {alert_type} alert (cooldown)")
                return
            
            sent_count = 0
            
            # Console alert (enhanced version for buy signals)
            if self.send_console_alert(title, message, alert_type):
                sent_count += 1
            
            # Desktop notification (additional to console)
            if self.config['desktop']['enabled']:
                if self.send_desktop_notification(title, message):
                    pass  # Don't count twice
            
            # File alert
            if self.send_file_alert(title, message):
                sent_count += 1
            
            # Email
            if self.config['email']['enabled']:
                if self.send_email_alert(f"AVAX Strategy: {title}", message):
                    sent_count += 1
                    print("   ✅ Email sent")
            
            # Discord
            if self.config['discord']['enabled']:
                discord_msg = f"🚨 **{title}**\n{message}"
                if self.send_discord_alert(discord_msg):
                    sent_count += 1
                    print("   ✅ Discord sent")
            
            # Slack
            if self.config['slack']['enabled']:
                slack_msg = f"🚨 *{title}*\n{message}"
                if self.send_slack_alert(slack_msg):
                    sent_count += 1
                    print("   ✅ Slack sent")
            
            # Save to history
            self.save_history(alert_type, f"{title}: {message}")
            
            print(f"   📊 Alert sent via {sent_count} method(s)")
            
            return sent_count > 0
        
        except Exception as e:
            print(f"Send alert failed: {e}")
            return False
    
    def check_alerts(self, analysis):
        """Check if any alerts should be triggered - compatible with monitor"""
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
1. Check your trading platform
2. Execute trade if conditions still valid
3. Monitor position according to strategy rules
                """
                self.send_alert('buy_signal', 'AVAX BUY SIGNAL', message)
            
            # Confidence threshold
            elif confidence >= thresholds.get('confidence_above', 0.6):
                message = f"Confidence reached {confidence:.3f}\nPrice: ${price:.2f}\nRegime: {regime.trend.value}/{regime.volatility.value}"
                self.send_alert('confidence', 'High Confidence Detected', message)
            
            # Price alerts
            elif price <= thresholds.get('price_below', 0):
                message = f"Price dropped to ${price:.2f}\nBelow threshold: ${thresholds['price_below']:.2f}"
                self.send_alert('price_low', 'Price Alert - Low', message)
            
            elif price >= thresholds.get('price_above', 999):
                message = f"Price rose to ${price:.2f}\nAbove threshold: ${thresholds['price_above']:.2f}"
                self.send_alert('price_high', 'Price Alert - High', message)
            
            # Regime change detection
            if thresholds.get('regime_change', True):
                # Check if regime improved from last check
                if len(self.history) > 0:
                    last_alert = self.history[-1]
                    if 'strong_bear' in last_alert.get('message', '') and regime.trend.value != 'strong_bear':
                        message = f"Regime improved!\nNew: {regime.trend.value}/{regime.volatility.value}\nConfidence: {confidence:.3f}"
                        self.send_alert('regime_change', 'Market Regime Improved', message)
        
        except Exception as e:
            print(f"Check alerts failed: {e}")

def setup_alerts():
    """Interactive alert setup - compatible with monitor"""
    print("🔔 ALERT SYSTEM SETUP")
    print("=" * 30)
    
    alert_system = AlertSystem()
    config = alert_system.config
    
    # Desktop notifications (always enabled)
    print("✅ Console/Desktop notifications: Enabled")
    
    # Email setup
    print("\n📧 EMAIL ALERTS:")
    setup_email = input("Enable email alerts? (y/n): ").lower() == 'y'
    
    if setup_email:
        config['email']['enabled'] = True
        config['email']['sender_email'] = input("Your email (sender): ")
        config['email']['recipient_email'] = input("Alert recipient email: ")
        
        print("\n⚠️  For Gmail, use an App Password:")
        print("   1. Go to Google Account settings")
        print("   2. Security → App passwords")
        print("   3. Generate password for 'Mail'")
        config['email']['sender_password'] = input("Email password/app password: ")
        
        print("✅ Email alerts configured")
    
    # Discord setup
    print("\n💬 DISCORD ALERTS:")
    setup_discord = input("Enable Discord alerts? (y/n): ").lower() == 'y'
    
    if setup_discord:
        config['discord']['enabled'] = True
        print("Create a Discord webhook:")
        print("   1. Go to your Discord server")
        print("   2. Server Settings → Integrations → Webhooks")
        print("   3. Create webhook, copy URL")
        config['discord']['webhook_url'] = input("Discord webhook URL: ")
        print("✅ Discord alerts configured")
    
    # Slack setup
    print("\n💼 SLACK ALERTS:")
    setup_slack = input("Enable Slack alerts? (y/n): ").lower() == 'y'
    
    if setup_slack:
        config['slack']['enabled'] = True
        print("Create a Slack webhook:")
        print("   1. Go to api.slack.com/apps")
        print("   2. Create app → Incoming Webhooks")
        print("   3. Activate and create webhook")
        config['slack']['webhook_url'] = input("Slack webhook URL: ")
        print("✅ Slack alerts configured")
    
    # Alert thresholds
    print("\n🎯 ALERT THRESHOLDS:")
    
    try:
        confidence_thresh = input(f"Confidence alert threshold (current: {config['thresholds']['confidence_above']}): ")
        if confidence_thresh:
            config['thresholds']['confidence_above'] = float(confidence_thresh)
        
        price_low = input(f"Price low alert (current: ${config['thresholds']['price_below']}): ")
        if price_low:
            config['thresholds']['price_below'] = float(price_low)
        
        cooldown = input(f"Alert cooldown minutes (current: {config['cooldown_minutes']}): ")
        if cooldown:
            config['cooldown_minutes'] = int(cooldown)
    
    except ValueError:
        print("⚠️ Invalid input, using defaults")
    
    # Save configuration
    alert_system.save_config()
    
    print("\n🎉 ALERT SYSTEM CONFIGURED!")
    print("✅ Console/Desktop notifications: Always enabled")
    print(f"✅ Email alerts: {'Enabled' if config['email']['enabled'] else 'Disabled'}")
    print(f"✅ Discord alerts: {'Enabled' if config['discord']['enabled'] else 'Disabled'}")
    print(f"✅ Slack alerts: {'Enabled' if config['slack']['enabled'] else 'Disabled'}")
    
    # Test alerts
    test_alert = input("\nSend test alert? (y/n): ").lower() == 'y'
    if test_alert:
        alert_system.send_alert('test', 'AVAX Strategy Test', 'Alert system is working!')
    
    return alert_system

def test_alert_system():
    """Test the alert system"""
    alert_system = AlertSystem()
    
    # Create fake analysis for testing
    class FakeRegime:
        def __init__(self):
            self.trend = type('obj', (object,), {'value': 'mild_bull'})
            self.volatility = type('obj', (object,), {'value': 'normal'})
            self.momentum = type('obj', (object,), {'value': 'steady_up'})
            self.confidence = 0.65
    
    fake_analysis = {
        'timestamp': datetime.now(),
        'price': 18.50,
        'regime': FakeRegime(),
        'buy_level': 18.00,
        'sell_level': 22.00,
        'should_enter': True
    }
    
    print("🧪 Testing alert system with fake BUY signal...")
    alert_system.check_alerts(fake_analysis)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        setup_alerts()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_alert_system()
    else:
        print("Usage:")
        print("  python advanced_alerts.py setup  # Configure alerts")
        print("  python advanced_alerts.py test   # Test alerts")