from dataclasses import dataclass

@dataclass
class Phase1Params:
    risk_reward: float
    trend: float
    entry: float
    confidence: float

@dataclass
class Phase2Params(Phase1Params):
    bullish: float
    bearish: float
    top: float
    bottom: float
    neutral: float

@dataclass
class Phase3Params(Phase2Params):
    window: int
    volume: float
    price: float
    trend_weight: float
