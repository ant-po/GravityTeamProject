import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Optional, List

from concepts import PositionUpdate, Trade, OrderType, OrderSide, OrderBook, TradeReason, TradeCosts
from time_manager import TimeSlice
from configs import MarketConfig, TradingConfig, DataConfig
from period_cycle import CyclePrediction


@dataclass
class ActiveMakerOrder:
    side: OrderSide
    size: float
    price: float
    timestamp: pd.Timestamp
    base_price: float
    filled_size: float = 0.0
    last_fill_time: Optional[pd.Timestamp] = None
    entry_reason: Optional[TradeReason] = None
    exit_reason: Optional[TradeReason] = None

    def __post_init__(self):
        if self.last_fill_time is None:
            self.last_fill_time = self.timestamp

    @property
    def is_complete(self) -> bool:
        return abs(self.filled_size - self.size) < 1e-10


@dataclass
class OrderTypeDecision:
    order_type: OrderType
    price_impact: float
    fill_time: float
    modified_size: float
    fee: float
    slippage: float


class TradeCostCalculator:
    def __init__(self, market_config: MarketConfig):
        self.market_config = market_config

    def calculate_taker_costs(self, size: float, side: OrderSide) -> TradeCosts:
        """
        Calculate taker trade costs
        """
        # base taker fee
        fee_impact = self.market_config.TAKER_FEE_BPS / 10000

        # half of spread for crossing book
        spread_impact = self.market_config.SPREAD_BPS / 20000

        # calculate slippage based on size
        slippage_steps = size / self.market_config.TAKER_SLIPPAGE_EXPOSURE_STEP
        slippage = slippage_steps * self.market_config.TAKER_SLIPPAGE_INCREASE_STEP_BPS / 10000

        # total price impact
        total_impact = fee_impact + spread_impact + slippage

        # adjust sign based on side
        if side == OrderSide.SELL:
            total_impact = -total_impact
            slippage = -slippage
            fee_impact = -fee_impact

        return TradeCosts(
            price_impact=total_impact,
            fee_component=fee_impact,
            slippage_component=slippage,
            fill_time=0
        )

    def calculate_maker_costs(self, size: float, side: OrderSide) -> TradeCosts:
        """
        Calculate maker trade costs
        """
        # maker fee
        fee_impact = self.market_config.MAKER_FEE_BPS / 10000

        # half spread to get back to best bid
        spread_impact = -self.market_config.SPREAD_BPS / 20000

        # top of book adjustment
        top_of_book_impact = self.market_config.MAKER_TOP_OF_BOOK_ADJ_BPS / 10000

        # total price impact
        total_impact = spread_impact + fee_impact + top_of_book_impact

        # adjust sign based on side
        if side == OrderSide.SELL:
            total_impact = -total_impact
            fee_impact = -fee_impact

        return TradeCosts(
            price_impact=total_impact,
            fee_component=fee_impact,
            slippage_component=0.0,  # No slippage for maker orders
            fill_time=(size / self.market_config.MAKER_FILL_EXPOSURE_STEP) * self.market_config.MAKER_FILL_SECONDS_STEP
        )

    @staticmethod
    def calculate_execution_price(base_price: float, costs: TradeCosts) -> float:
        """
        Calculate final execution price including all costs
        """
        return base_price * (1 + costs.price_impact)


@dataclass
class PositionState:
    """
    Track state of current position and trade management
    """
    is_active: bool = False
    entry_price: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None
    size: float = 0.0
    stop_loss_price: Optional[float] = None
    expected_peak_index: Optional[int] = None
    current_prediction: Optional[CyclePrediction] = None


class PositionManager:
    def __init__(self,
                 trading_config: TradingConfig,
                 market_config: MarketConfig,
                 data_config: DataConfig):
        self.trading_config = trading_config
        self.market_config = market_config
        self.data_config = data_config
        self.stop_loss_pct = trading_config.STOP_LOSS_PCT
        self.state = PositionState()
        self.cost_calculator = TradeCostCalculator(market_config)

    def update_position(self, trade):
        """
        Update current position after trade execution
        """
        if not self.state.is_active:
            # new position being initiated
            self.state.size = trade.size * trade.side.value
            self.state.is_active = True
            return

        # update existing position size
        self.state.size += trade.size * trade.side.value

        # check if position closed
        if abs(self.state.size) < 1e-10:
            self.state = PositionState()  # Resets is_active to False

    def calculate_position_size_order_type(self,
                                           prediction: CyclePrediction,
                                           current_idx: int
                                           ) -> (float, OrderTypeDecision):
        """
        Calculate optimal position size considering execution constraints
        """
        print(f"Calculating position size for prediction at idx {current_idx}")
        print(f"Expected trough-to-peak return: {prediction.trough_to_peak_return:.6f}")

        # get total available time for trade
        total_time_to_peak = (prediction.time_to_trough + prediction.time_trough_to_peak -
                              self.data_config.MIN_POINTS_IN_PERIOD)

        # try different sizes to find optimal considering both time and costs
        test_sizes = np.linspace(0, self.trading_config.MAX_POSITION, 20)
        best_size = 0.0
        best_net_return = 0.0
        best_order_decision = None

        for size in test_sizes:
            # get maker metrics
            maker_costs = self.cost_calculator.calculate_maker_costs(size, OrderSide.BUY)
            taker_costs = self.cost_calculator.calculate_taker_costs(size, OrderSide.BUY)

            # if maker time is too long, force taker order
            if maker_costs.fill_time > total_time_to_peak:
                total_costs = taker_costs.price_impact
                order_type = OrderType.TAKER
                fill_time = taker_costs.fill_time
                slippage = taker_costs.slippage_component
                fee = taker_costs.fee_component
            else:
                if maker_costs.price_impact <= taker_costs.price_impact:
                    total_costs = maker_costs.price_impact
                    order_type = OrderType.MAKER
                    fill_time = maker_costs.fill_time
                    slippage = maker_costs.slippage_component
                    fee = maker_costs.fee_component
                else:
                    total_costs = taker_costs.price_impact
                    order_type = OrderType.TAKER
                    fill_time = taker_costs.fill_time
                    slippage = taker_costs.slippage_component
                    fee = taker_costs.fee_component

            # calculate expected return for this size
            net_return = prediction.trough_to_peak_return * size - total_costs

            if net_return > best_net_return:
                best_net_return = net_return
                best_size = size
                best_order_decision = OrderTypeDecision(
                    order_type=order_type,
                    price_impact=total_costs,
                    fill_time=fill_time,
                    modified_size=size,
                    fee=fee,
                    slippage=slippage
                )

        # ensure minimum profitability
        min_acceptable_return = 0.00001  # 1bp minimum acceptable return
        if best_net_return < min_acceptable_return:
            print(f"Trade cannot be placed - costs & time outweigh expected return")
            return 0.0, None

        print(f"Selected size {best_size:.4f} with net return {best_net_return:.6f}")
        print(f"Order type: {best_order_decision.order_type}")
        print(f"Price impact: {best_order_decision.price_impact:.6f}")

        return best_size, best_order_decision


class TradeManager:
    def __init__(self,
                 market_config: MarketConfig,
                 trading_config: TradingConfig
                 ):
        """
        Initialise the trade executor
        """
        self.market_config = market_config
        self.trading_config = trading_config
        self.cost_calculator = TradeCostCalculator(market_config)
        self.active_maker_order: Optional[ActiveMakerOrder] = None
        self.trades: List[Trade] = []

    def process_time_slice(self, time_slice: TimeSlice) -> Optional[Trade]:
        if not self.active_maker_order:
            return None

        print(f"Processing maker order at {time_slice.timestamp}")

        # check if enough time has passed since last fill
        if (time_slice.timestamp - self.active_maker_order.last_fill_time).total_seconds() < \
                self.market_config.MAKER_FILL_SECONDS_STEP:
            return None

        # calculate fill amount
        remaining_size = self.active_maker_order.size - self.active_maker_order.filled_size
        fill_amount = min(self.market_config.MAKER_FILL_EXPOSURE_STEP, remaining_size)

        if fill_amount > 0:
            # create and record trade

            costs = self.cost_calculator.calculate_maker_costs(fill_amount, self.active_maker_order.side)
            trade = Trade(
                side=self.active_maker_order.side,
                size=fill_amount,
                price=self.active_maker_order.price,
                timestamp=time_slice.timestamp,
                fees=costs.fee_component,
                slippage=costs.slippage_component,
                base_price=self.active_maker_order.base_price,
                order_type=OrderType.MAKER,
                entry_reason=self.active_maker_order.entry_reason,
                exit_reason=self.active_maker_order.exit_reason
            )

            # update order progress
            self.active_maker_order.filled_size += fill_amount
            self.active_maker_order.last_fill_time = time_slice.timestamp

            self.trades.append(trade)
            print(f"Maker fill completed: {fill_amount} at {trade.price}")

            # clear completed order
            if self.active_maker_order.is_complete:
                self.active_maker_order = None

            return trade

        return None

    def execute_trade(self,
                      position_update: PositionUpdate,
                      order_book: OrderBook,
                      order_type_decision: OrderTypeDecision,
                      current_idx: int,
                      entry_reason: Optional[TradeReason] = None,
                      exit_reason: Optional[TradeReason] = None) -> Optional[Trade]:
        """
        Execute trade based on position update and order type decision
        """
        # prevent any new trades if there's an active maker order
        if self.active_maker_order is not None:
            return None

        costs = self.cost_calculator.calculate_taker_costs(position_update.size, position_update.side)
        execution_price = self.cost_calculator.calculate_execution_price(order_book.mid, costs)

        if order_type_decision.order_type == OrderType.TAKER:
            trade = Trade(
                side=position_update.side,
                size=position_update.size,
                price=execution_price,
                timestamp=position_update.timestamp,
                fees=costs.fee_component,
                slippage=costs.slippage_component,
                base_price=order_book.mid,
                order_type=order_type_decision.order_type,
                entry_reason=entry_reason if position_update.side == OrderSide.BUY else None,
                exit_reason=exit_reason if position_update.side == OrderSide.SELL else None
            )
            self.trades.append(trade)
            return trade
        else:
            costs = self.cost_calculator.calculate_taker_costs(position_update.size, position_update.side)
            execution_price = self.cost_calculator.calculate_execution_price(order_book.mid, costs)

            self.active_maker_order = ActiveMakerOrder(
                side=position_update.side,
                size=position_update.size,
                price=execution_price,
                timestamp=position_update.timestamp,
                base_price=order_book.mid,
                filled_size=0.0,
                last_fill_time=None,
                entry_reason=entry_reason if position_update.side == OrderSide.BUY else None,
                exit_reason=exit_reason if position_update.side == OrderSide.SELL else None
            )
            return None

    def get_trades(self) -> List[Trade]:
        """
        Get list of all trades
        """
        return self.trades.copy()
