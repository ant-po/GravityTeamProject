import pandas as pd

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Period:
    # basic period info
    type: Literal['under', 'over']
    start_idx: int
    end_idx: int

    # price info at boundaries
    start_price: float
    start_spread: float
    end_price: float
    end_spread: float

    # track extremes
    extreme_price: float  # lowest for under, highest for over
    extreme_idx: int
    extreme_spread: float

    # return metrics
    price_return: float  # Total return over period
    return_to_extreme: float  # Return from start to extreme
    return_from_extreme: float  # Return from extreme to end

    # spread metrics
    start_to_extreme_spread: float  # Spread change from start to extreme
    extreme_to_end_spread: float  # Spread change from extreme to end

    # duration
    duration: int  # end_idx - start_idx + 1

    # stats used for similarity matching
    early_mid_return: float
    early_mid_volatility: float
    early_spread_change: float


def identify_all_periods(series_a: pd.Series,
                         series_b: pd.Series,
                         min_consecutive_points: int = 5,
                         idx_start_ref: int = 0) -> [Period]:
    """
    Identify both under and over periods where series_a is consistently below/above series_b.
    Calculate comprehensive metrics for each period.
    """
    spread = series_a - series_b
    n = len(spread)
    periods = []

    def create_period(start: int, end: int, type_: str) -> Optional[Period]:
        if end - start + 1 < min_consecutive_points:
            return None

        # extract period data
        period_mid= series_a[start:end + 1]
        period_spread = spread[start:end + 1]

        # calculate early period metrics
        early_window = min_consecutive_points
        early_mid = series_a[start:start + early_window]
        early_spread = spread[start:start + early_window]

        # early return
        early_mid_return = (early_mid.iloc[-1] / early_mid.iloc[0]) - 1

        # early volatility (std of returns)
        early_returns = early_mid.pct_change().dropna()
        early_mid_volatility = early_returns.std()

        # early spread change
        early_spread_change = early_spread.iloc[-1] - early_spread.iloc[0]

        # find extreme point (min for under, max for over)
        if type_ == 'under':
            extreme_idx_rel = period_mid.argmin()
            extreme_price = period_mid.iloc[extreme_idx_rel]
        else:
            extreme_idx_rel = period_mid.argmax()
            extreme_price = period_mid.iloc[extreme_idx_rel]

        extreme_idx = start + extreme_idx_rel
        extreme_spread = period_spread.iloc[extreme_idx_rel]

        # calculate returns
        start_price = period_mid.iloc[0]
        end_price = period_mid.iloc[-1]
        price_return = (end_price / start_price) - 1
        return_to_extreme = (extreme_price / start_price) - 1
        return_from_extreme = (end_price / extreme_price) - 1

        # calculate spread changes
        start_spread = period_spread.iloc[0]
        end_spread = period_spread.iloc[-1]
        start_to_extreme_spread = extreme_spread - start_spread
        extreme_to_end_spread = end_spread - extreme_spread

        return Period(
            type=type_,
            start_idx=start + idx_start_ref,
            end_idx=end + idx_start_ref,
            start_price=start_price,
            start_spread=start_spread,
            end_price=end_price,
            end_spread=end_spread,
            extreme_price=extreme_price,
            extreme_idx=extreme_idx + idx_start_ref,
            extreme_spread=extreme_spread,
            price_return=price_return,
            return_to_extreme=return_to_extreme,
            return_from_extreme=return_from_extreme,
            start_to_extreme_spread=start_to_extreme_spread,
            extreme_to_end_spread=extreme_to_end_spread,
            duration=end - start + 1,
            early_mid_return=early_mid_return,
            early_mid_volatility=early_mid_volatility,
            early_spread_change=early_spread_change
        )

    # handle single period case first
    is_under = spread.iloc[0] < 0
    if all(s < 0 for s in spread) or all(s > 0 for s in spread):
        period = create_period(0, n - 1, 'under' if is_under else 'over')
        if period:
            periods.append(period)
        return periods

    # state tracking
    current_streak = 0
    potential_reversal_idx = None
    reversal_streak = 0
    current_type = 'under' if is_under else 'over'
    period_start = 0

    for i in range(n):
        is_under = spread.iloc[i] < 0
        is_current_under = current_type == 'under'

        # point matches current period type
        if is_under == is_current_under:
            current_streak += 1
            if potential_reversal_idx is not None:
                potential_reversal_idx = None
                reversal_streak = 0

        # point differs from current period type
        else:
            # first point of potential reversal
            if potential_reversal_idx is None:
                potential_reversal_idx = i
                reversal_streak = 1
            else:
                reversal_streak += 1

            # check if we have enough points to confirm reversal
            if reversal_streak >= min_consecutive_points:
                # add previous period if it was long enough
                if current_streak >= min_consecutive_points:
                    period = create_period(period_start, potential_reversal_idx - 1, current_type)
                    if period:
                        periods.append(period)

                # start new period
                period_start = potential_reversal_idx
                current_type = 'under' if is_under else 'over'
                current_streak = reversal_streak
                potential_reversal_idx = None
                reversal_streak = 0

    # handle final period
    if potential_reversal_idx is None:
        if current_streak >= min_consecutive_points:
            period = create_period(period_start, n - 1, current_type)
            if period:
                periods.append(period)
    elif reversal_streak >= min_consecutive_points:
        if current_streak >= min_consecutive_points:
            period = create_period(period_start, potential_reversal_idx - 1, current_type)
            if period:
                periods.append(period)
        if reversal_streak >= min_consecutive_points:
            period = create_period(potential_reversal_idx, n - 1, 'under' if current_type == 'over' else 'over')
            if period:
                periods.append(period)
    elif current_streak >= min_consecutive_points:
        period = create_period(period_start, n - 1, current_type)
        if period:
            periods.append(period)

    return periods


def start_of_last_under_period(periods: [Period]) -> (int, bool):
    if periods[-1].type == 'over':
        return periods[-2].start_idx, True
    else:
        return periods[-1].start_idx, False


@dataclass
class Cycle:
    # start to trough characteristics
    start_idx: int
    trough_idx: int
    start_to_trough_return: float

    # trough to peak characteristics
    peak_idx: int
    trough_to_peak_return: float

    # peak to end characteristics
    end_idx: int
    peak_to_end_return: float

    # overall metrics
    total_return: float
    duration: int

    # stats used for similarity matching
    early_mid_return: float = None
    early_mid_volatility: float = None
    early_spread_change: float = None


def construct_cycles(periods: [Period]) -> [Cycle]:
    """
    Construct cycles from alternating under/over periods.
    Each cycle consists of an under period followed by an over period.
    """
    cycles = []

    if periods[0].type == 'over':
        periods = periods[1:]

    # process pairs of periods
    for i in range(0, len(periods) - 1, 2):
        under_period = periods[i]
        over_period = periods[i + 1]

        # validate period sequence
        if under_period.type != 'under' or over_period.type != 'over':
            print(f"Warning: Invalid period sequence at index {i}. Skipping cycle.")
            continue

        # create cycle
        cycle = Cycle(
            # start to trough
            start_idx=under_period.start_idx,
            trough_idx=under_period.extreme_idx,
            start_to_trough_return=under_period.return_to_extreme,

            # trough to peak
            peak_idx=over_period.extreme_idx,
            trough_to_peak_return=(over_period.extreme_price / under_period.extreme_price) - 1,

            # peak to end
            end_idx=over_period.end_idx,
            peak_to_end_return=over_period.return_from_extreme,

            # overall metrics
            total_return=(over_period.end_price / under_period.start_price) - 1,
            duration=over_period.end_idx - under_period.start_idx + 1,

            # stats used for similarity matching
            early_mid_return=under_period.early_mid_return,
            early_mid_volatility=under_period.early_mid_volatility,
            early_spread_change=under_period.early_spread_change
        )

        cycles.append(cycle)

    if periods[-1].type == 'under':
        under_period = periods[-1]
        # if still seeking trough (extreme point is at end)
        if under_period.extreme_idx == under_period.end_idx:
            peak_idx = None
            trough_to_peak_return = None
        else:
            # have trough, use current price if higher than trough as temp peak
            if under_period.end_price > under_period.extreme_price:
                peak_idx = under_period.end_idx
                trough_to_peak_return = (under_period.end_price / under_period.extreme_price) - 1
            else:
                peak_idx = under_period.extreme_idx
                trough_to_peak_return = 0

        cycle = Cycle(
            start_idx=under_period.start_idx,
            trough_idx=under_period.extreme_idx,
            start_to_trough_return=under_period.return_to_extreme,
            peak_idx=peak_idx,
            trough_to_peak_return=trough_to_peak_return,
            end_idx=under_period.end_idx,
            peak_to_end_return=under_period.return_from_extreme,
            total_return=(under_period.end_price / under_period.start_price) - 1,
            duration=under_period.duration,
            early_mid_return=under_period.early_mid_return,
            early_mid_volatility=under_period.early_mid_volatility,
            early_spread_change=under_period.early_spread_change
        )
        cycles.append(cycle)

    return cycles


def analyze_cycle_characteristics(cycles: [Cycle]) -> pd.DataFrame:
    """
    Create summary statistics for identified cycles.
    """
    metrics = []
    for cycle in cycles:
        metrics.append({
            'Duration': cycle.duration,
            'Start to Trough Time': cycle.trough_idx - cycle.start_idx,
            'Trough to Peak Time': cycle.peak_idx - cycle.trough_idx,
            'Peak to End Time': cycle.end_idx - cycle.peak_idx,
            'Start to Trough Return': cycle.start_to_trough_return,
            'Trough to Peak Return': cycle.trough_to_peak_return,
            'Peak to End Return': cycle.peak_to_end_return,
            'Total Return': cycle.total_return
        })

    df = pd.DataFrame(metrics)
    return df.describe()

@dataclass
class CyclePrediction:
    time_to_trough: int
    time_trough_to_peak: int
    trough_to_peak_return: float
    confidence_score: float


class CycleTracker:
    def __init__(self, min_consecutive_points: int = 5):
        self.min_consecutive_points = min_consecutive_points

    def update(self,
               mid_series: pd.Series,
               reference_series: pd.Series,
               idx_start_ref: int
               ) -> [Cycle]:
        """
        Update cycle tracking with latest data.
        Always processes data from start of last known 'under' period.
        """

        # get periods and construct cycles
        periods = identify_all_periods(mid_series, reference_series, self.min_consecutive_points, idx_start_ref)
        cycles = construct_cycles(periods)

        return cycles, periods
