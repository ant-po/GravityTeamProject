import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass
from typing import Literal

from concepts import OrderBook
from configs import MarketConfig

HEADERS_MAPPING = {
    'gt_timestamp': 'timestamp',
    'exch_rate': 'reference',
    'executable_rate': 'mid'
}


class DataProcessor:
    def __init__(self, file_path: str|Path, training_period_ratio: float = 0.5,
                 headers_mapping: dict[str, str] = None,
                 timestamp_fmt: str = '%Y-%m-%d %H:%M:%S.%f'):
        """
        Initialise the data processor
        """
        self.training_period_ratio = training_period_ratio

        self.file_path = Path(file_path).resolve()
        self.headers_mapping = headers_mapping
        if self.headers_mapping is None:
            self.headers_mapping = HEADERS_MAPPING
        self.timestamp_fmt = timestamp_fmt

    def load_and_prepare_data(self) -> (pd.Series, pd.Series, pd.Series):
        """
        Load and prepare the data
        """
        # load data
        file_type = self.file_path.suffix.lower()
        match file_type:
            case '.csv':
                raw_data = pd.read_csv(self.file_path)
            case '.parquet':
                raw_data = pd.read_parquet(self.file_path)
            case _:
                raise ValueError(f'Cannot load file {str(self.file_path)}')
        # adjust headers and set timestamp as index
        data = raw_data.rename(columns=self.headers_mapping)
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], format=self.timestamp_fmt)
        data.set_index('timestamp', inplace=True)
        data.sort_index(inplace=True)

        # add spread column
        data['spread'] = data['mid'] - data['reference']

        print("Sample of loaded data:")
        print(f"Mid prices range: {data['mid'].min():.6f} to {data['mid'].max():.6f}")
        print(f"Reference prices range: {data['reference'].min():.6f} to {data['reference'].max():.6f}")
        print(f"Spread range: {data['spread'].min():.6f} to {data['spread'].max():.6f}")

        return data['mid'], data['reference'], data['spread']


class OrderBookFeed:
    def __init__(self, market_config: MarketConfig):
        self.bidask_spread_bps = market_config.SPREAD_BPS

    @staticmethod
    def get_best_bid_ask(mid: float, spread_bps: float) -> (float, float):
        """
        Calculate bid and ask prices given the mid price and spread (assuming symmetric spread)
        """
        spread_decimal = spread_bps / 10000
        half_spread = spread_decimal / 2

        bid = mid * (1 - half_spread)
        ask = mid * (1 + half_spread)

        return bid, ask

    def update(self, mid: float) -> OrderBook:
        ob = OrderBook()
        ob.mid = mid
        ob.bid, ob.ask = self.get_best_bid_ask(mid, self.bidask_spread_bps)
        return ob


@dataclass
class PeriodMetrics:
    period_type: Literal['over', 'under']
    start_idx: int
    end_idx: int
    duration: int
    # Deviation peaks
    peak_deviation: float
    peak_deviation_idx: int
    peak_deviation_pct: float
    # Return peaks
    peak_return: float
    peak_return_idx: int
    peak_return_pct: float
    # Returns
    cum_return_to_peak_a: float
    cum_return_after_peak_a: float
    total_return_a: float
    total_return_b: float


def analyze_series_periods(series_a: pd.Series, series_b: pd.Series) -> [PeriodMetrics]:
    """
    Analyze periods where series A is above or below series B with both deviation and return peaks
    """
    # Calculate difference and returns
    diff = series_a - series_b
    returns_a = series_a.pct_change().fillna(0)
    returns_b = series_b.pct_change().fillna(0)

    # Find crossover points
    sign_changes = np.where(np.diff(np.signbit(diff)))[0]

    # If no crossovers, treat entire period as one
    if len(sign_changes) == 0:
        start_idx = 0
        end_indices = [len(diff) - 1]
    else:
        # Add start and end indices if needed
        if sign_changes[0] != 0:
            sign_changes = np.insert(sign_changes, 0, 0)
        if sign_changes[-1] != len(diff) - 1:
            sign_changes = np.append(sign_changes, len(diff) - 1)

        start_idx = 0
        end_indices = sign_changes[1:]

    periods = []

    # Analyze each period
    for end_idx in end_indices:
        period_slice = slice(start_idx, end_idx + 1)
        period_diff = diff[period_slice]

        # Determine period type based on average position
        avg_diff = period_diff.mean()
        period_type = 'over' if avg_diff > 0 else 'under'

        # Find deviation peak
        abs_diff = np.abs(period_diff)
        dev_peak_relative_idx = abs_diff.argmax()
        peak_deviation = period_diff.iloc[dev_peak_relative_idx]
        peak_deviation_idx = start_idx + dev_peak_relative_idx

        # Calculate returns within the period
        period_prices_a = series_a.iloc[period_slice]
        period_prices_b = series_b.iloc[period_slice]
        cum_returns_a = (period_prices_a / period_prices_a.iloc[0] - 1)

        # Find return peak
        if period_type == 'over':
            ret_peak_relative_idx = cum_returns_a.argmax()
        else:
            ret_peak_relative_idx = cum_returns_a.argmin()

        peak_return = cum_returns_a.iloc[ret_peak_relative_idx]
        peak_return_idx = start_idx + ret_peak_relative_idx

        # Calculate timing metrics
        duration = len(period_diff)

        # Calculate percentage positions for both peaks
        peak_deviation_pct = (dev_peak_relative_idx / (duration - 1) * 100
                              if duration > 1 else 100)
        peak_return_pct = (ret_peak_relative_idx / (duration - 1) * 100
                           if duration > 1 else 100)

        # Calculate returns
        start_price_a = period_prices_a.iloc[0]
        end_price_a = period_prices_a.iloc[-1]
        peak_price_a = period_prices_a.iloc[ret_peak_relative_idx]

        start_price_b = period_prices_b.iloc[0]
        end_price_b = period_prices_b.iloc[-1]

        # Calculate return metrics
        cum_return_to_peak_a = (peak_price_a / start_price_a) - 1
        cum_return_after_peak_a = (end_price_a / peak_price_a) - 1 if ret_peak_relative_idx < len(
            period_diff) - 1 else 0
        total_return_a = (end_price_a / start_price_a) - 1
        total_return_b = (end_price_b / start_price_b) - 1

        periods.append(PeriodMetrics(
            period_type=period_type,
            start_idx=start_idx,
            end_idx=end_idx,
            duration=duration,
            peak_deviation=peak_deviation,
            peak_deviation_idx=peak_deviation_idx,
            peak_deviation_pct=peak_deviation_pct,
            peak_return=peak_return,
            peak_return_idx=peak_return_idx,
            peak_return_pct=peak_return_pct,
            cum_return_to_peak_a=cum_return_to_peak_a,
            cum_return_after_peak_a=cum_return_after_peak_a,
            total_return_a=total_return_a,
            total_return_b=total_return_b
        ))

        start_idx = end_idx

    return periods


def analyze_period_transitions(periods: [PeriodMetrics]) -> dict:
    """
    Analyze transitions between periods to find non-alternating sequences

    Args:
        periods: List of period metrics
    """
    if len(periods) <= 1:
        return {
            'total_periods': len(periods),
            'non_alternating_count': 0,
            'non_alternating_ratio': 0.0,
            'sequence_details': []
        }

    non_alternating_count = 0
    sequence_details = []

    # Check each adjacent pair of periods
    for i in range(len(periods) - 1):
        current_period = periods[i]
        next_period = periods[i + 1]

        if current_period.period_type == next_period.period_type:
            non_alternating_count += 1
            sequence_details.append({
                'index': i,
                'type': current_period.period_type,
                'start_idx': current_period.start_idx,
                'end_idx': next_period.end_idx
            })

    return {
        'total_periods': len(periods),
        'non_alternating_count': non_alternating_count,
        'non_alternating_ratio': non_alternating_count / (len(periods) - 1),
        'sequence_details': sequence_details
    }


def create_period_summary(periods: [PeriodMetrics],
                          output_path: Path) -> pd.DataFrame:
    """Create and save summary table of period metrics"""
    summary_data = []
    for period in periods:
        summary_data.append({
            'Type': period.period_type,
            'Start Index': period.start_idx,
            'End Index': period.end_idx,
            'Duration (points)': period.duration,
            'Peak Deviation': period.peak_deviation,
            'Peak Deviation Index': period.peak_deviation_idx,
            'Peak Deviation Position (%)': period.peak_deviation_pct,
            'Peak Return': period.peak_return,
            'Peak Return Index': period.peak_return_idx,
            'Peak Return Position (%)': period.peak_return_pct,
            'Return to Peak A (%)': period.cum_return_to_peak_a * 100,
            'Return after Peak A (%)': period.cum_return_after_peak_a * 100,
            'Total Return A (%)': period.total_return_a * 100,
            'Total Return B (%)': period.total_return_b * 100,
            'Return Difference (A-B) (%)': (period.total_return_a - period.total_return_b) * 100
        })

    df = pd.DataFrame(summary_data)
    df.to_csv(output_path, index=False)
    return df


def plot_duration_vs_returns(periods: [PeriodMetrics],
                             output_path: Path):
    """Create scatter plot of period durations vs returns with regression lines"""
    plt.figure(figsize=(12, 8))

    # Separate over and under periods
    over_periods = [p for p in periods if p.period_type == 'over']
    under_periods = [p for p in periods if p.period_type == 'under']

    # Plot and fit 'over' periods
    if over_periods:
        durations_over = np.array([p.duration for p in over_periods])
        returns_over = np.array([p.cum_return_to_peak_a * 100 for p in over_periods])

        # Plot scatter
        plt.scatter(durations_over, returns_over, c='green', label='Over Periods', alpha=0.6)

        # Fit regression line
        if len(durations_over) > 1:  # Need at least 2 points for regression
            z = np.polyfit(durations_over, returns_over, 1)
            p = np.poly1d(z)
            plt.plot(durations_over, p(durations_over), "g--",
                     alpha=0.8, label=f'Over Trend: {z[0]:.2e}x + {z[1]:.2e}')

    # Plot and fit 'under' periods
    if under_periods:
        durations_under = np.array([p.duration for p in under_periods])
        returns_under = np.array([p.cum_return_to_peak_a * 100 for p in under_periods])

        # Plot scatter
        plt.scatter(durations_under, returns_under, c='red', label='Under Periods', alpha=0.6)

        # Fit regression line
        if len(durations_under) > 1:  # Need at least 2 points for regression
            z = np.polyfit(durations_under, returns_under, 1)
            p = np.poly1d(z)
            plt.plot(durations_under, p(durations_under), "r--",
                     alpha=0.8, label=f'Under Trend: {z[0]:.2e}x + {z[1]:.2e}')

    plt.xlabel('Period Duration (data points)')
    plt.ylabel('Cumulative Return to Peak (%)')
    plt.title('Period Duration vs Returns')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(output_path)
    plt.close()


def plot_metric_distributions(periods: [PeriodMetrics], output_dir: Path):
    """
    Create histogram plots for key metrics, separated by period type

    Args:
        periods: List of period metrics
        output_dir: Directory to save plots
    """
    # Create metrics to plot
    metrics_to_plot = {
        'duration': {
            'data': lambda p: p.duration,
            'title': 'Distribution of Period Durations',
            'xlabel': 'Duration (points)',
            'filename': 'duration_distribution.png'
        },
        'dev_peak_pos': {
            'data': lambda p: p.peak_deviation_pct,
            'title': 'Distribution of Peak Deviation Positions',
            'xlabel': 'Position in Period (%)',
            'filename': 'peak_deviation_position_distribution.png'
        },
        'ret_peak_pos': {
            'data': lambda p: p.peak_return_pct,
            'title': 'Distribution of Peak Return Positions',
            'xlabel': 'Position in Period (%)',
            'filename': 'peak_return_position_distribution.png'
        },
        'ret_to_peak': {
            'data': lambda p: p.cum_return_to_peak_a * 100,
            'title': 'Distribution of Returns to Peak',
            'xlabel': 'Return (%)',
            'filename': 'return_to_peak_distribution.png'
        },
        'ret_after_peak': {
            'data': lambda p: p.cum_return_after_peak_a * 100,
            'title': 'Distribution of Returns after Peak',
            'xlabel': 'Return (%)',
            'filename': 'return_after_peak_distribution.png'
        }
    }

    # Separate periods by type
    over_periods = [p for p in periods if p.period_type == 'over']
    under_periods = [p for p in periods if p.period_type == 'under']

    # Create and save each histogram
    for metric_name, metric_info in metrics_to_plot.items():
        plt.figure(figsize=(12, 6))

        # Create subplot for better organization
        plt.subplot(1, 2, 1)
        if over_periods:
            data_over = [metric_info['data'](p) for p in over_periods]
            plt.hist(data_over, bins=20, color='green', alpha=0.6, label='Over')
            plt.axvline(np.mean(data_over), color='green', linestyle='dashed',
                        linewidth=2, label=f'Mean: {np.mean(data_over):.2f}')
        plt.title('Over Periods')
        plt.xlabel(metric_info['xlabel'])
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(1, 2, 2)
        if under_periods:
            data_under = [metric_info['data'](p) for p in under_periods]
            plt.hist(data_under, bins=20, color='red', alpha=0.6, label='Under')
            plt.axvline(np.mean(data_under), color='red', linestyle='dashed',
                        linewidth=2, label=f'Mean: {np.mean(data_under):.2f}')
        plt.title('Under Periods')
        plt.xlabel(metric_info['xlabel'])
        plt.ylabel('Frequency')
        plt.legend()

        plt.suptitle(metric_info['title'])
        plt.tight_layout()

        # Save plot
        plt.savefig(output_dir / metric_info['filename'])
        plt.close()


def analyze_and_visualise(series_a: pd.Series,
                          series_b: pd.Series,
                          output_dir: Path) -> ([PeriodMetrics], pd.DataFrame, dict):
    """
    Perform complete analysis and create visualizations
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze periods
    periods = analyze_series_periods(series_a, series_b)

    # Analyze transitions
    transition_stats = analyze_period_transitions(periods)

    # Create and save summary table
    summary_df = create_period_summary(periods, output_dir / 'period_summary.csv')

    # Create and save duration vs returns plot
    plot_duration_vs_returns(periods, output_dir / 'duration_vs_returns.png')

    # Create and save distribution plots
    plot_metric_distributions(periods, output_dir)

    # Save transition statistics
    transition_summary = pd.DataFrame([{
        'Total Periods': transition_stats['total_periods'],
        'Non-alternating Sequences': transition_stats['non_alternating_count'],
        'Non-alternating Ratio': transition_stats['non_alternating_ratio'],
        'Over Periods Count': len([p for p in periods if p.period_type == 'over']),
        'Under Periods Count': len([p for p in periods if p.period_type == 'under'])
    }])
    transition_summary.to_csv(output_dir / 'transition_stats.csv', index=False)

    return periods, summary_df, transition_stats
