import pandas as pd

from enum import Enum
from dataclasses import dataclass
from typing import Optional

from data_analysis import CyclePrediction

class OrderSide(Enum):
    BUY = 1
    SELL = -1


@dataclass
class OrderBook:
    # state of the order book at particular timestamp
    mid: float = 0.0  # mid-price
    bid: float = 0.0 # best bid
    ask: float = 0.0 # best ask


class OrderType(Enum):
    MAKER = "MAKER"
    TAKER = "TAKER"


@dataclass
class Position:
    size: float  # Current position size in base currency
    timestamp: pd.Timestamp


@dataclass
class PositionUpdate:
    size: float
    side: OrderSide
    timestamp: pd.Timestamp


class TradeReason(Enum):
    # Entry reason
    CYCLE_ENTRY = "CYCLE_ENTRY"  # Entry at predicted trough

    # Exit reasons
    CYCLE_PEAK_EXIT = "CYCLE_PEAK_EXIT"  # Exit at predicted peak
    CYCLE_STOP_LOSS = "CYCLE_STOP_LOSS"  # Stop loss hit
    CYCLE_TIME_EXIT = "CYCLE_TIME_EXIT"  # Maximum hold time reached
    CYCLE_NEW_UNDER = "CYCLE_NEW_UNDER"


@dataclass
class Trade:
    side: OrderSide
    size: float
    price: float  # final execution price, including all costs
    timestamp: pd.Timestamp
    fees: float = 0.0  # fees paid
    slippage: float = 0.0  # slippage incurred
    base_price: float = 0.0  # original price before fees/slippage
    order_type: OrderType = OrderType.TAKER
    entry_reason: Optional[TradeReason] = None
    exit_reason: Optional[TradeReason] = None
    entry_metrics: Optional[dict] = None  # Store metrics at entry
    exit_metrics: Optional[dict] = None   # Store metrics at exit


@dataclass
class TradeCosts:
    price_impact: float  # Total price impact as a decimal
    fee_component: float  # Fee component of price impact
    slippage_component: float  # Slippage component of price impact
    fill_time: float  # time to fill the order


@dataclass
class Signal:
    target: float  # 0 to 1 (trade size ranges 0% to 100% of available notional)
    conviction: float  # 0 to 1 (confidence in signal ranges 0% to 100%, can be used to derive appropriate size)
    timestamp: pd.Timestamp
    prediction: Optional[CyclePrediction] = None
    expected_peak_index: Optional[int] = None
