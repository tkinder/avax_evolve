# historical_trade_simulator.py
import pandas as pd
import sys
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append('src')

from core.regime_detector import EnhancedRegimeDetector, RegimeBasedTradingRules
from core.adaptive_backtester import AdaptiveParams, AdaptivePriceLevels

class HistoricalTradeSimulator:
    """
    Simulate what would have happened with historical trades
    """
    
    def __init__(self):
        # Load the hybrid data
        self.df = pd.read_csv('hybrid_avax_data.csv', index_col=0, parse_dates=True)
        
        # Your optimized parameters
        self.params = AdaptiveParams(
            buy_threshold_pct=0.162,
            sell_threshold_pct=0.691,
            regime_short_window=10,
            regime_medium_window=30,
            regime_long_window=60,
            max_position_pct=0.80,
            stop_loss_pct=0.082,        # 8.2%
            take_profit_pct=0.150,      # 15.0%
        )
        
        # Initialize components
        self.regime_detector = EnhancedRegimeDetector(
            short_window=self.params.regime_short_window,
            medium_window=self.params.regime_medium_window,
            long_window=self.params.regime_long_window
        )
        
        self.price_calculator = AdaptivePriceLevels(self.params)
        
        print("📊 Historical Trade Simulator Ready")
        print(f"📈 Data: {len(self.df)} days ({self.df.index[0].strftime('%Y-%m-%d')} to {self.df.index[-1].strftime('%Y-%m-%d')})")
    
    def simulate_june_5_trade(self):
        """
        Simulate what would have happened if you bought on June 5
        """
        print("\n🎯 SIMULATING JUNE 5 TRADE")
        print("=" * 40)
        
        # Find June 5 in data
        try:
            entry_date = '2025-06-05'
            # Fix the date lookup method
            entry_idx = None
            for i, date in enumerate(self.df.index):
                if date.strftime('%Y-%m-%d') == entry_date:
                    entry_idx = i
                    break
            
            if entry_idx is None:
                print(f"❌ Could not find {entry_date} in data")
                return
            
            entry_price = self.df.iloc[entry_idx]['close']
            
            print(f"📅 Entry: {entry_date}")
            print(f"💰 Entry Price: ${entry_price:.2f}")
            
            # Calculate stop loss and take profit
            stop_loss_price = entry_price * (1 - self.params.stop_loss_pct)
            take_profit_price = entry_price * (1 + self.params.take_profit_pct)
            
            print(f"🛡️  Stop Loss: ${stop_loss_price:.2f} (-8.2%)")
            print(f"🎯 Take Profit: ${take_profit_price:.2f} (+15.0%)")
            
            # Simulate holding the position day by day
            position_active = True
            current_idx = entry_idx
            days_held = 0
            
            print(f"\n📈 DAILY SIMULATION:")
            print("Date       Price   P&L%    Days  Regime                     Exit Reason")
            print("-" * 80)
            
            while position_active and current_idx < len(self.df) - 1:
                current_idx += 1
                days_held += 1
                current_date = self.df.index[current_idx]
                current_price = self.df.iloc[current_idx]['close']
                
                # Calculate P&L
                unrealized_pnl = (current_price - entry_price) / entry_price
                
                # Get current regime
                regime = self.regime_detector.detect_regime(self.df, current_idx)
                
                # Check exit conditions using your enhanced logic
                should_exit, exit_reason = self.check_enhanced_exit_conditions(
                    regime, current_price, entry_price, days_held, current_idx
                )
                
                # Print daily status
                regime_str = f"{regime.trend.value[:4]}/{regime.volatility.value[:3]}/{regime.momentum.value[:6]}"
                print(f"{current_date.strftime('%m-%d')} ${current_price:7.2f} {unrealized_pnl:6.1%} {days_held:5d}  {regime_str:20} {exit_reason if should_exit else ''}")
                
                if should_exit:
                    print(f"\n🚪 EXIT TRIGGERED!")
                    print(f"📅 Exit Date: {current_date.strftime('%Y-%m-%d')}")
                    print(f"💰 Exit Price: ${current_price:.2f}")
                    print(f"📊 Total Return: {unrealized_pnl:.1%}")
                    print(f"📅 Days Held: {days_held}")
                    print(f"🔍 Exit Reason: {exit_reason}")
                    
                    # Calculate actual loss/gain
                    if unrealized_pnl > 0:
                        print("✅ Trade would have been PROFITABLE")
                    else:
                        print("❌ Trade would have been a LOSS")
                    
                    position_active = False
                
                # Safety check - don't run forever
                if days_held > 30:
                    print("⏰ Simulation stopped at 30 days")
                    break
            
            if position_active:
                # Position still active at end of data
                final_price = self.df.iloc[-1]['close']
                final_pnl = (final_price - entry_price) / entry_price
                
                print(f"\n📊 POSITION STILL ACTIVE (end of data)")
                print(f"💰 Current Price: ${final_price:.2f}")
                print(f"📊 Unrealized P&L: {final_pnl:.1%}")
                
        except Exception as e:
            print(f"❌ Simulation error: {e}")
    
    def check_enhanced_exit_conditions(self, regime, current_price, entry_price, days_held, current_idx):
        """
        Check your enhanced exit conditions
        """
        unrealized_pnl = (current_price - entry_price) / entry_price
        
        # Get adaptive price levels
        buy_level, sell_level, support, resistance = self.price_calculator.get_price_levels(self.df, current_idx)
        
        # Your enhanced exit logic (from adaptive_backtester.py)
        
        # 1. Standard stop loss
        if unrealized_pnl <= -self.params.stop_loss_pct:
            return True, "stop_loss"
        
        # 2. Standard take profit
        if current_price >= sell_level or unrealized_pnl >= self.params.take_profit_pct:
            return True, "take_profit"
        
        # 3. Regime emergency
        if regime.should_avoid_trading():
            return True, "regime_emergency"
        
        # 4. Bear market exit
        if regime.trend.value == "strong_bear":
            return True, "bear_market_exit"
        
        # 5. Early bear exit (more conservative)
        if unrealized_pnl < -0.06 or (unrealized_pnl < -0.02 and days_held > 7):
            return True, "early_bear_exit"
        
        # 6. Momentum exit
        if (unrealized_pnl < -0.04 and days_held >= 3 and 
            regime.momentum.value == "accel_down"):
            return True, "momentum_exit"
        
        # 7. Volatility exit
        if regime.volatility.value == "extreme" and unrealized_pnl > 0.05:
            return True, "volatility_exit"
        
        return False, ""
    
    def analyze_all_recent_signals(self):
        """
        Analyze all recent potential signals
        """
        print("\n🔍 ANALYZING ALL RECENT SIGNALS")
        print("=" * 45)
        
        # Look for signals in last 30 days
        start_idx = len(self.df) - 30
        signals = []
        
        for i in range(start_idx, len(self.df)):
            try:
                regime = self.regime_detector.detect_regime(self.df, i)
                buy_level, sell_level, support, resistance = self.price_calculator.get_price_levels(self.df, i)
                current_price = self.df['close'].iloc[i]
                
                # Check if this would have been a signal
                basic_signal = RegimeBasedTradingRules.should_enter_long(regime, current_price, buy_level)
                confidence_ok = regime.confidence >= 0.6
                
                if basic_signal and confidence_ok:
                    signals.append({
                        'date': self.df.index[i],
                        'price': current_price,
                        'regime': regime,
                        'buy_level': buy_level,
                        'sell_level': sell_level,
                        'confidence': regime.confidence
                    })
            except:
                continue
        
        print(f"🎯 Found {len(signals)} potential signals in last 30 days:")
        
        for signal in signals:
            print(f"\n📅 {signal['date'].strftime('%Y-%m-%d')}:")
            print(f"   💰 Price: ${signal['price']:.2f}")
            print(f"   🧠 Regime: {signal['regime'].trend.value}/{signal['regime'].volatility.value}/{signal['regime'].momentum.value}")
            print(f"   🎯 Confidence: {signal['confidence']:.3f}")
            print(f"   📈 Buy Level: ${signal['buy_level']:.2f}")
            print(f"   📉 Target: ${signal['sell_level']:.2f}")
            
            # Quick analysis of what happened after
            signal_date_str = signal['date'].strftime('%Y-%m-%d')
            signal_idx = None
            for i, date in enumerate(self.df.index):
                if date.strftime('%Y-%m-%d') == signal_date_str:
                    signal_idx = i
                    break
            
            if signal_idx and signal_idx < len(self.df) - 5:  # At least 5 days of data after
                prices_after = self.df['close'].iloc[signal_idx:signal_idx+6]  # Next 5 days
                max_gain = (prices_after.max() - signal['price']) / signal['price']
                max_loss = (prices_after.min() - signal['price']) / signal['price']
                final_5day = (prices_after.iloc[-1] - signal['price']) / signal['price']
                
                print(f"   📊 Next 5 days: Max gain {max_gain:.1%}, Max loss {max_loss:.1%}, Final {final_5day:.1%}")

if __name__ == "__main__":
    simulator = HistoricalTradeSimulator()
    simulator.simulate_june_5_trade()
    simulator.analyze_all_recent_signals()