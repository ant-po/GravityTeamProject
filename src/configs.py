from dataclasses import dataclass


@dataclass(frozen=True)
class MarketConfig:
    # maker order parameters
    MAKER_TOP_OF_BOOK_ADJ_BPS: float = 1.0  # we assume that maker order is placed at the top of the book
    MAKER_FILL_EXPOSURE_STEP: float = 0.01  # 1% exposure
    MAKER_FILL_SECONDS_STEP: float = 3.0  # every 3 seconds
    MAKER_FEE_BPS: float = 0.0

    # taker order parameters
    TAKER_FEE_BPS: float = 5.0
    TAKER_SLIPPAGE_INCREASE_STEP_BPS: float = 1  # 1bp increase in slippage
    TAKER_SLIPPAGE_EXPOSURE_STEP: float = 0.02  # per 2% exposure

    # order book parameters
    SPREAD_BPS: float = 10.0  # constant spread


@dataclass(frozen=True)
class TradingConfig:
    # strategy parameters
    WINDOW = 200

    # position limits
    MAX_POSITION: float = 1.0
    MIN_POSITION: float = 0.0  # we assume no shorting

    # simulation parameters
    INITIAL_CAPITAL: float = 1000.0
    STOP_LOSS_PCT: float = 0.02


@dataclass(frozen=True)
class DataConfig:
    TRAINING_POINTS: int = 750000
    MIN_CONSECUTIVE_POINTS: int = 5
    MIN_POINTS_IN_PERIOD: int = 20