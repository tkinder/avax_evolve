# src/core/regime_detector.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from enum import Enum

class TrendRegime(Enum):
    STRONG_BULL = "strong_bull"
    MILD_BULL = "mild_bull" 
    SIDEWAYS = "sideways"
    MILD_BEAR = "mild_bear"
    STRONG_BEAR = "strong_bear"

class VolatilityRegime(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

class MomentumRegime(Enum):
    ACCELERATING_UP = "accel_up"
    STEADY_UP = "steady_up"
    STALLING = "stalling"
    STEADY_DOWN = "steady_down"
    ACCELERATING_DOWN = "accel_down"

@dataclass
class MarketRegime:
    trend: TrendRegime
    volatility: VolatilityRegime
    momentum: MomentumRegime
    confidence: float  # 0-1 confidence in the regime classification
    
    def __str__(self):
        return f"{self.trend.value}/{self.volatility.value}/{self.momentum.value}"
    
    def is_bullish(self) -> bool:
        return self.trend in [TrendRegime.STRONG_BULL, TrendRegime.MILD_BULL]
    
    def is_bearish(self) -> bool:
        return self.trend in [TrendRegime.STRONG_BEAR, TrendRegime.MILD_BEAR]
    
    def is_high_volatility(self) -> bool:
        return self.volatility in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]
    
    def should_avoid_trading(self) -> bool:
        """High-risk conditions where trading should be avoided."""
        return (
            self.trend == TrendRegime.STRONG_BEAR or
            self.volatility == VolatilityRegime.EXTREME or
            (self.is_bearish() and self.is_high_volatility()) or
            self.confidence < 0.3
        )

class EnhancedRegimeDetector:
    """
    Advanced market regime detection using multiple indicators.
    """
    
    def __init__(self, 
                 short_window: int = 10,
                 medium_window: int = 30, 
                 long_window: int = 60):
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
    
    def detect_regime(self, df: pd.DataFrame, current_idx: int) -> MarketRegime:
        """
        Detect current market regime using multiple indicators.
        """
        if current_idx < self.long_window:
            # Not enough data - assume neutral
            return MarketRegime(
                trend=TrendRegime.SIDEWAYS,
                volatility=VolatilityRegime.NORMAL,
                momentum=MomentumRegime.STALLING,
                confidence=0.2
            )
        
        # Get recent data
        start_idx = max(0, current_idx - self.long_window)
        recent_data = df.iloc[start_idx:current_idx + 1]
        
        # Detect each regime component
        trend_regime = self._detect_trend_regime(recent_data)
        volatility_regime = self._detect_volatility_regime(recent_data)
        momentum_regime = self._detect_momentum_regime(recent_data)
        confidence = self._calculate_confidence(recent_data, trend_regime, volatility_regime)
        
        return MarketRegime(
            trend=trend_regime,
            volatility=volatility_regime,
            momentum=momentum_regime,
            confidence=confidence
        )
    
    def _detect_trend_regime(self, data: pd.DataFrame) -> TrendRegime:
        """
        Detect trend using multiple timeframes and indicators.
        """
        prices = data['close'].values
        
        # Multiple moving averages
        short_ma = np.mean(prices[-self.short_window:])
        medium_ma = np.mean(prices[-self.medium_window:])
        long_ma = np.mean(prices[-self.long_window:])
        
        # Price vs moving averages
        current_price = prices[-1]
        
        # Trend strength indicators
        short_vs_medium = (short_ma - medium_ma) / medium_ma
        medium_vs_long = (medium_ma - long_ma) / long_ma
        price_vs_long = (current_price - long_ma) / long_ma
        
        # Linear regression slope
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        slope_pct = slope / np.mean(prices) * len(prices)  # Normalize slope
        
        # Combine indicators for trend classification
        trend_score = (
            0.3 * np.sign(short_vs_medium) * min(abs(short_vs_medium) * 10, 1) +
            0.3 * np.sign(medium_vs_long) * min(abs(medium_vs_long) * 10, 1) +
            0.2 * np.sign(price_vs_long) * min(abs(price_vs_long) * 5, 1) +
            0.2 * np.sign(slope_pct) * min(abs(slope_pct) * 2, 1)
        )
        
        # Classify trend
        if trend_score > 0.6:
            return TrendRegime.STRONG_BULL
        elif trend_score > 0.2:
            return TrendRegime.MILD_BULL
        elif trend_score > -0.2:
            return TrendRegime.SIDEWAYS
        elif trend_score > -0.6:
            return TrendRegime.MILD_BEAR
        else:
            return TrendRegime.STRONG_BEAR
    
    def _detect_volatility_regime(self, data: pd.DataFrame) -> VolatilityRegime:
        """
        Detect volatility regime using rolling statistics.
        """
        returns = data['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return VolatilityRegime.NORMAL
        
        # Current volatility (recent period)
        recent_vol = returns.tail(self.short_window).std()
        
        # Historical volatility (full period)
        historical_vol = returns.std()
        
        # Volatility ratio
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
        
        # Average true range proxy
        high_low_range = (data['close'].rolling(self.short_window).max() - 
                         data['close'].rolling(self.short_window).min()).iloc[-1]
        avg_price = data['close'].tail(self.short_window).mean()
        range_pct = high_low_range / avg_price if avg_price > 0 else 0
        
        # Combine volatility indicators
        vol_score = 0.6 * vol_ratio + 0.4 * (range_pct * 10)
        
        # Classify volatility
        if vol_score > 3.0:
            return VolatilityRegime.EXTREME
        elif vol_score > 1.8:
            return VolatilityRegime.HIGH
        elif vol_score > 0.6:
            return VolatilityRegime.NORMAL
        else:
            return VolatilityRegime.LOW
    
    def _detect_momentum_regime(self, data: pd.DataFrame) -> MomentumRegime:
        """
        Detect momentum regime using rate of change analysis.
        """
        prices = data['close'].values
        
        if len(prices) < self.medium_window:
            return MomentumRegime.STALLING
        
        # Rate of change over different periods
        short_roc = (prices[-1] - prices[-self.short_window]) / prices[-self.short_window]
        medium_roc = (prices[-1] - prices[-self.medium_window]) / prices[-self.medium_window]
        
        # Acceleration (change in rate of change)
        if len(prices) >= self.medium_window + self.short_window:
            prev_short_roc = (prices[-self.short_window] - prices[-self.short_window*2]) / prices[-self.short_window*2]
            acceleration = short_roc - prev_short_roc
        else:
            acceleration = 0
        
        # Moving average convergence/divergence proxy
        short_ma = np.mean(prices[-self.short_window:])
        medium_ma = np.mean(prices[-self.medium_window:])
        
        if len(prices) >= self.medium_window + 5:
            prev_short_ma = np.mean(prices[-self.short_window-5:-5])
            prev_medium_ma = np.mean(prices[-self.medium_window-5:-5])
            
            current_convergence = short_ma - medium_ma
            prev_convergence = prev_short_ma - prev_medium_ma
            convergence_change = current_convergence - prev_convergence
        else:
            convergence_change = 0
        
        # Combine momentum indicators
        momentum_score = (
            0.4 * short_roc * 10 +
            0.3 * medium_roc * 10 +
            0.2 * acceleration * 50 +
            0.1 * convergence_change / np.mean(prices) * 100
        )
        
        # Classify momentum
        if momentum_score > 0.5:
            return MomentumRegime.ACCELERATING_UP
        elif momentum_score > 0.1:
            return MomentumRegime.STEADY_UP
        elif momentum_score > -0.1:
            return MomentumRegime.STALLING
        elif momentum_score > -0.5:
            return MomentumRegime.STEADY_DOWN
        else:
            return MomentumRegime.ACCELERATING_DOWN
    
    def _calculate_confidence(self, data: pd.DataFrame, 
                            trend: TrendRegime, 
                            volatility: VolatilityRegime) -> float:
        """
        Calculate confidence in the regime classification.
        """
        prices = data['close'].values
        
        # Consistency of trend
        if len(prices) >= self.medium_window:
            recent_highs = len(prices[-self.short_window:][prices[-self.short_window:] > np.mean(prices[-self.medium_window:])])
            trend_consistency = abs(recent_highs - self.short_window/2) / (self.short_window/2)
        else:
            trend_consistency = 0
        
        # Volatility stability
        returns = data['close'].pct_change().dropna()
        if len(returns) >= self.short_window:
            vol_stability = 1 - min(returns.tail(self.short_window).std() / max(returns.std(), 0.001), 2) / 2
        else:
            vol_stability = 0.5
        
        # Data sufficiency
        data_sufficiency = min(len(data) / self.long_window, 1)
        
        # Combine confidence factors
        confidence = (
            0.4 * (1 - trend_consistency) +
            0.3 * vol_stability +
            0.3 * data_sufficiency
        )
        
        return max(0.1, min(1.0, confidence))

class RegimeBasedTradingRules:
    """
    Trading rules that adapt based on detected market regime.
    """
    
    @staticmethod
    def should_enter_long(regime: MarketRegime, current_price: float, buy_level: float) -> bool:
        """
        Determine if we should enter a long position based on regime.
        """
        # Never trade in high-risk conditions
        if regime.should_avoid_trading():
            return False
        
        # Basic price condition
        price_condition = current_price <= buy_level
        
        # Regime-specific conditions
        if regime.trend == TrendRegime.STRONG_BEAR:
            return False  # Never buy in strong bear market
        
        elif regime.trend == TrendRegime.MILD_BEAR:
            # Only buy on extreme oversold in mild bear
            return price_condition and regime.momentum in [MomentumRegime.STALLING]
        
        elif regime.trend == TrendRegime.SIDEWAYS:
            # Buy dips in sideways market
            return price_condition and regime.volatility != VolatilityRegime.EXTREME
        
        elif regime.trend in [TrendRegime.MILD_BULL, TrendRegime.STRONG_BULL]:
            # More aggressive buying in bull markets
            return price_condition
        
        return False
    
    @staticmethod
    def should_exit_long(regime: MarketRegime, current_price: float, entry_price: float, 
                        sell_level: float, stop_loss_pct: float, take_profit_pct: float) -> Tuple[bool, str]:
        """
        Determine if we should exit a long position based on regime.
        Returns (should_exit, reason)
        """
        unrealized_pnl = (current_price - entry_price) / entry_price
        
        # Emergency exit conditions
        if regime.should_avoid_trading():
            return True, "regime_emergency"
        
        # Standard stop loss
        if unrealized_pnl <= -stop_loss_pct:
            return True, "stop_loss"
        
        # Standard take profit  
        if current_price >= sell_level or unrealized_pnl >= take_profit_pct:
            return True, "take_profit"
        
        # Regime-specific exits
        if regime.trend == TrendRegime.STRONG_BEAR:
            return True, "bear_market_exit"
        
        elif regime.trend == TrendRegime.MILD_BEAR and unrealized_pnl < -stop_loss_pct * 0.5:
            return True, "early_bear_exit"
        
        elif regime.volatility == VolatilityRegime.EXTREME and unrealized_pnl > 0:
            return True, "volatility_exit"  # Take profits in extreme volatility
        
        elif regime.momentum == MomentumRegime.ACCELERATING_DOWN and unrealized_pnl < 0:
            return True, "momentum_exit"
        
        return False, ""
    
    @staticmethod
    def get_position_size_multiplier(regime: MarketRegime) -> float:
        """
        Get position size multiplier based on regime.
        """
        base_multiplier = 1.0
        
        # Trend adjustments
        if regime.trend == TrendRegime.STRONG_BULL:
            base_multiplier *= 1.3
        elif regime.trend == TrendRegime.MILD_BULL:
            base_multiplier *= 1.1
        elif regime.trend == TrendRegime.MILD_BEAR:
            base_multiplier *= 0.7
        elif regime.trend == TrendRegime.STRONG_BEAR:
            base_multiplier *= 0.3  # Very small positions
        
        # Volatility adjustments
        if regime.volatility == VolatilityRegime.EXTREME:
            base_multiplier *= 0.4
        elif regime.volatility == VolatilityRegime.HIGH:
            base_multiplier *= 0.7
        elif regime.volatility == VolatilityRegime.LOW:
            base_multiplier *= 1.2
        
        # Confidence adjustments
        base_multiplier *= regime.confidence
        
        return max(0.1, min(1.5, base_multiplier))

# Test function
def test_regime_detection():
    """
    Test the enhanced regime detection system.
    """
    from src.core.data import fetch_historical_data
    
    print("🧪 Testing Enhanced Regime Detection...")
    
    df = fetch_historical_data(refresh=False)
    detector = EnhancedRegimeDetector()
    
    # Test on key periods
    test_dates = [
        ('2023-06-01', 'Bull market period'),
        ('2024-03-01', 'Peak period'),
        ('2024-08-01', 'Bear market period'),
        ('2025-03-01', 'Recent period')
    ]
    
    for date_str, description in test_dates:
        try:
            date_idx = df.index.get_loc(date_str, method='nearest')
            regime = detector.detect_regime(df, date_idx)
            
            print(f"\n📅 {description} ({date_str}):")
            print(f"   Regime: {regime}")
            print(f"   Confidence: {regime.confidence:.2f}")
            print(f"   Should avoid trading: {regime.should_avoid_trading()}")
            print(f"   Position multiplier: {RegimeBasedTradingRules.get_position_size_multiplier(regime):.2f}x")
            
        except Exception as e:
            print(f"   ❌ Error testing {date_str}: {e}")

if __name__ == "__main__":
    test_regime_detection()