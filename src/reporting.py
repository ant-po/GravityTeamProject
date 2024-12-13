import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

from concepts import Trade, OrderType, TradeReason
from data_processing import OrderBookFeed
from configs import MarketConfig, TradingConfig
from period_cycle import Cycle, CyclePrediction


@dataclass
class CycleComparison:
    predicted_trough_to_peak_return: float
    actual_trough_to_peak_return: float
    prediction_error: float  # Difference between predicted and actual
    exit_reason: str


class CyclePerformanceTracker:
    def __init__(self, market_config):
        self.comparisons: List[CycleComparison] = []
        self.current_prediction: Optional[CyclePrediction] = None
        self.entry_price: Optional[float] = None
        self.peak_price: Optional[float] = None

        # track trades
        self.trades: List[Trade] = []
        self.performances = self.comparisons

    def record_trade(self, trade: Trade):
        """
        Record a trade execution
        """
        self.trades.append(trade)

    def start_new_cycle(self, prediction: CyclePrediction, entry_price: float):
        """
        Record the start of a new cycle with its prediction
        """
        self.current_prediction = prediction
        self.entry_price = entry_price
        self.peak_price = entry_price  # Initialise peak tracking

    def update_peak_price(self, current_price: float):
        """
        Update the highest price seen during the cycle
        """
        if current_price > self.peak_price:
            self.peak_price = current_price

    def complete_cycle(self, current_cycle: Cycle, exit_reason: TradeReason):
        """
        Compare prediction with actual cycle performance
        """
        if not self.current_prediction or not self.entry_price:
            return

        # calculate actual trough-to-peak return
        actual_return = (self.peak_price / self.entry_price) - 1

        comparison = CycleComparison(
            predicted_trough_to_peak_return=self.current_prediction.trough_to_peak_return,
            actual_trough_to_peak_return=actual_return,
            prediction_error=actual_return - self.current_prediction.trough_to_peak_return,
            exit_reason=exit_reason.name
        )

        self.comparisons.append(comparison)

        # reset state
        self.current_prediction = None
        self.entry_price = None
        self.peak_price = None

    def generate_performance_report(self) -> str:
        """
        Generate a simple report comparing predictions to actuals
        """
        if not self.comparisons:
            return "No completed cycles to report."

        prediction_errors = [c.prediction_error for c in self.comparisons]
        avg_error = sum(prediction_errors) / len(prediction_errors)

        report = f"""
Cycle Performance Report
-----------------------
Number of completed cycles: {len(self.comparisons)}
Average prediction error: {avg_error:.2%}

Exit Statistics:
{self._get_exit_statistics()}
"""
        return report

    def _get_exit_statistics(self) -> str:
        """
        Calculate statistics about exit reasons
        """
        exit_counts = {}
        for comp in self.comparisons:
            exit_counts[comp.exit_reason] = exit_counts.get(comp.exit_reason, 0) + 1

        stats = []
        total = len(self.comparisons)
        for reason, count in exit_counts.items():
            percentage = (count / total) * 100
            stats.append(f"- {reason}: {percentage:.1f}%")

        return "\n".join(stats)


class PerformanceAnalyser:
    def __init__(self):
        pass

    @staticmethod
    def calculate_pnl(trades: List[Trade], mid: pd.Series, market_config: MarketConfig, trading_config: TradingConfig) -> (pd.Series, pd.Series):
        """
        Calculate PnL series from trades
        """
        order_book_feed = OrderBookFeed(market_config)
        pnl = pd.Series(0.0, index=mid.index)
        returns = pd.Series(0.0, index=mid.index)
        initial_capital = trading_config.INITIAL_CAPITAL  # Starting capital
        usdt_available = initial_capital
        token_position = 0.0  # Units of token held
        previous_portfolio_value = initial_capital
        position = 0.0

        for timestamp in mid.index:
            order_book = order_book_feed.update(mid[timestamp])

            # process any trades (includes partial fills)
            trades_at_time = [t for t in trades if t.timestamp == timestamp]
            for trade in trades_at_time:
                usdt_amount = trade.size * previous_portfolio_value  # Amount of USDT to trade
                token_amount = usdt_amount / trade.price  # Convert to token units
                token_position += token_amount * trade.side.value  # Get tokens
                usdt_available -= usdt_amount * trade.side.value  # Pay USDT
                position += trade.size * trade.side.value

            # calculate portfolio_value immediately after the trade
            portfolio_value_post_trade = usdt_available + token_position * order_book.bid
            pnl.loc[timestamp] = portfolio_value_post_trade - previous_portfolio_value
            previous_portfolio_value = portfolio_value_post_trade
            returns.loc[timestamp] = pnl.loc[timestamp] / previous_portfolio_value  # Convert to percentage

        return pnl, returns

    @staticmethod
    def calculate_metrics(pnl: pd.Series, returns: pd.Series, trades: List[Trade]) -> Dict[str, float]:
        """
        Calculate performance metrics including detailed fee and slippage analysis
        """
        total_pnl = pnl.sum()
        cum_pnl = pnl.cumsum()

        # separate maker and taker trades
        maker_trades = [t for t in trades if t.order_type == OrderType.MAKER]
        taker_trades = [t for t in trades if t.order_type == OrderType.TAKER]

        metrics = {
            'total_pnl': total_pnl,
            'total_return': returns.sum(),
            'max_drawdown': (cum_pnl - cum_pnl.cummax()).min(),
            'num_trades': len(trades),
            'num_maker_trades': len(maker_trades),
            'num_taker_trades': len(taker_trades),
            'win_rate': len([t for t in trades if t.price * t.size * t.side.value > t.fees]) / len(
                trades) if trades else 0,
            'avg_trade_size': np.mean([t.size for t in trades]) if trades else 0,

            # slippage metrics
            'avg_slippage': np.mean([t.slippage for t in trades]) if trades else 0,
            'avg_maker_slippage': np.mean([t.slippage for t in maker_trades]) if maker_trades else 0,
            'avg_taker_slippage': np.mean([t.slippage for t in taker_trades]) if taker_trades else 0,

            # fee metrics
            'total_fees': sum(t.fees for t in trades),
            'total_maker_fees': sum(t.fees for t in maker_trades),
            'total_taker_fees': sum(t.fees for t in taker_trades),
            'avg_fee_per_trade': np.mean([t.fees for t in trades]) if trades else 0,
        }

        return metrics

    def generate_report(self, pnl: pd.Series, returns: pd.Series, trades: List[Trade]) -> str:
        """
        Generate a detailed performance report including execution metrics
        """
        metrics = self.calculate_metrics(pnl, returns, trades)

        report = f"""
Performance Report
-----------------
Total PnL: {metrics['total_pnl']:.2f} USDT
Total Return: {metrics['total_return']:.2f}%
Max Drawdown: {metrics['max_drawdown']:.2f} USDT

Trade Statistics
---------------
Number of Trades: {metrics['num_trades']} (Maker: {metrics['num_maker_trades']}, Taker: {metrics['num_taker_trades']})
Win Rate: {metrics['win_rate']:.2%}
Average Trade Size: {metrics['avg_trade_size']:.2f}

Execution Metrics
----------------
Average Slippage: {metrics['avg_slippage']:.6f}
  - Maker Trades: {metrics['avg_maker_slippage']:.6f}
  - Taker Trades: {metrics['avg_taker_slippage']:.6f}

Fee Analysis
-----------
Total Fees: {metrics['total_fees']:.2f} USDT
  - Maker Fees: {metrics['total_maker_fees']:.2f} USDT
  - Taker Fees: {metrics['total_taker_fees']:.2f} USDT
Average Fee per Trade: {metrics['avg_fee_per_trade']:.4f} USDT
        """

        return report


class PerformanceVisualiser:
    def __init__(self, figsize: tuple = (15, 8), style: str = 'whitegrid'):
        """
        Initialise the visualiser
        """
        self.figsize = figsize
        sns.set_style(style)

    @staticmethod
    def _calculate_position_series(trades: List[Trade], index: pd.Index) -> pd.Series:
        """
        Calculate position series for a specific period
        """
        position_series = pd.Series(0.0, index=index)
        current_position = 0.0

        for trade in trades:
            current_position += trade.size * trade.side.value
            position_series.loc[trade.timestamp:] = current_position

        return position_series

    def create_performance_plots(self,
                                pnl: pd.Series,
                                trades: List[Trade],
                                price_data: pd.DataFrame,
                                plot_dir: Path):
        """
        Create and save performance plots
        """

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)

        # plot 1: Cumulative PnL
        cumulative_pnl = pnl.cumsum()
        ax1.plot(cumulative_pnl.index, cumulative_pnl.values, 'b-', label='Cumulative PnL')
        ax1.set_title('Cumulative PnL')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('PnL (USDT)')
        ax1.grid(True)

        # plot 2: Position Size
        positions = self._calculate_position_series(trades, pnl.index)
        ax2.step(positions.index, positions.values, 'g-', label='Position Size')
        ax2.set_title('Position Size Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Position Size')
        ax2.grid(True)

        # adjust layout and save
        plt.tight_layout()
        plt.savefig(plot_dir / 'performance_plots.png')
        plt.close()
