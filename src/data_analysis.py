import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path


@dataclass
class UnderOverCycle:
    # Under period characteristics
    under_start_idx: int
    under_trough_idx: int
    under_end_idx: int
    under_duration: int
    trough_return: float
    trough_to_under_end_return: float

    # Over period characteristics
    over_start_idx: int
    over_peak_idx: int
    over_end_idx: int
    over_duration: int
    peak_return: float

    # Full cycle metrics
    trough_to_peak_return: float
    trough_to_peak_duration: int

    # Early indicators for prediction
    early_price_velocity: float
    early_spread_momentum: float
    early_return_volatility: float
    early_period_duration: int  # Duration of under period so far

    def __post_init__(self):
        # Validate that all durations are non-negative
        assert self.under_duration >= 0, "Under duration must be non-negative"
        assert self.over_duration >= 0, "Over duration must be non-negative"
        assert self.trough_to_peak_duration >= 0, "Trough to peak duration must be non-negative"
        assert self.early_period_duration >= 0, "Early period duration must be non-negative"


@dataclass
class CyclePrediction:
    expected_remaining_time_to_trough: int
    expected_time_from_trough_to_under_end: int
    expected_time_from_under_end_to_peak: int
    expected_trough_return: float
    expected_trough_to_under_end_return: float
    expected_under_end_to_peak_return: float
    expected_total_return: float
    confidence_score: float


@dataclass
class PeriodState:
    """
    Tracks the state and key metrics of a single under/over period
    """
    type: Literal['under', 'over']
    start_idx: int
    start_price: float
    start_spread: float

    # These are updated as period progresses
    current_idx: int = None
    current_price: float = None
    current_spread: float = None

    # Track extremes
    extreme_price: float = None  # lowest for under, highest for over
    extreme_idx: int = None

    # These are set when period ends
    end_idx: Optional[int] = None
    end_price: Optional[float] = None
    end_spread: Optional[float] = None
    duration: Optional[int] = None

    def update(self, idx: int, price: float, spread: float) -> None:
        """Update current state of period"""
        self.current_idx = idx
        self.current_price = price
        self.current_spread = spread

        # Update extreme based on period type
        if self.extreme_price is None:
            self.extreme_price = price
            self.extreme_idx = idx
        elif (self.type == 'under' and price < self.extreme_price) or \
             (self.type == 'over' and price > self.extreme_price):
            self.extreme_price = price
            self.extreme_idx = idx

        self.duration = idx - self.start_idx

    def complete(self, idx: int, price: float, spread: float) -> None:
        """
        Mark period as complete with final values
        """
        self.end_idx = idx
        self.end_price = price
        self.end_spread = spread
        self.duration = self.end_idx - self.start_idx

    @property
    def price_return(self) -> float:
        """
        Calculate total return for the period
        """
        if self.current_price is None:
            return 0.0
        return (self.current_price / self.start_price) - 1

    def __str__(self) -> str:
        return (f"{self.type.upper()} Period: "
                f"idx [{self.start_idx} -> {self.current_idx}], "
                f"spread [{self.start_spread:.6f} -> {self.current_spread:.6f}], "
                f"return {self.price_return:.4%}")


class CycleAnalysis:
    def __init__(self,
                 database
                 ):
        self.database = database


class CycleAnalyzer:
    def __init__(self,
                 early_period_window: int = 20,
                 num_shifts: int = 5,
                 shift_step: float = 0.1):
        self.early_period_window = early_period_window
        self.num_shifts = num_shifts
        self.shift_step = shift_step
        self.cycles_database: [UnderOverCycle] = []
        self.trading_reference: Optional[pd.Series] = None
        self.selected_shift: Optional[float] = None

    def generate_shifted_references(self, series_a: pd.Series, series_b: pd.Series) -> ([pd.Series, float]):
        """
        Generate shifted versions of reference series
        """
        spread = series_a - series_b
        typical_spread = np.median(abs(spread))

        shifts = np.linspace(-self.shift_step * self.num_shifts,
                             self.shift_step * self.num_shifts,
                             2 * self.num_shifts + 1)

        shifted_series = []
        for shift in shifts:
            shift_amount = typical_spread * shift
            shifted = series_b + shift_amount
            shifted_series.append((shifted, shift))

        return shifted_series

    def select_trading_reference(self, series_a: pd.Series, series_b: pd.Series) -> pd.Series:
        """
        Analyze all shifts and select median-cycle reference
        """
        shift_analysis = []
        all_cycles = []

        shifted_references = self.generate_shifted_references(series_a, series_b)

        for ref_series, shift in shifted_references:
            cycles = self.identify_cycles(series_a, ref_series)
            total_return = 0
            for cycle in cycles:
                cycle.shift_level = shift
                total_return += cycle.trough_to_peak_return

            all_cycles.extend(cycles)
            shift_analysis.append({
                'shift': shift,
                'num_cycles': len(cycles),
                'reference': ref_series,
                'cycles': cycles,
                'total_return': total_return
            })

        shift_analysis.sort(key=lambda x: x['total_return'])
        median_idx = len(shift_analysis) // 2
        # selected = shift_analysis[median_idx]
        selected = shift_analysis[-1]

        self.cycles_database = all_cycles
        self.trading_reference = selected['reference']
        self.selected_shift = selected['shift']

        return self.trading_reference


    def _calculate_return(self, series: pd.Series, window: int = 20) -> float:
        """
        Calculate momentum of price movements
        """
        returns = series.pct_change().fillna(0)
        if len(returns) < window:
            return 0.0
        return returns.sum()

    def _calculate_spread_momentum(self, series_a: pd.Series, series_b: pd.Series, window: int = 20) -> float:
        """
        Calculate momentum of the spread
        """
        spread = series_a - series_b
        if len(spread) < window:
            return 0.0
        return (spread.iloc[-1] - spread.iloc[-window]) / spread.iloc[-window] if abs(
            spread.iloc[-window]) > 1e-10 else 0.0

    def _calculate_volatility(self, series: pd.Series, window: int = 20) -> float:
        """Calculate rolling volatility of returns"""
        returns = series.pct_change().fillna(0)
        if len(returns) < window:
            return 0.0
        vol = returns.rolling(window=window).std().iloc[-1]
        return 0.0 if np.isnan(vol) else vol

    def _find_similar_cycles(self, current_metrics: dict[str, float], n_similar: int = 5) -> [UnderOverCycle]:
        """
        Find similar cycles using normalised metrics
        """
        print(f"Finding similar cycles from database of {len(self.cycles_database)} cycles")
        print(f"Current metrics before normalization: {current_metrics}")

        # Collect all metric values for normalization
        all_metrics = {
            'price_velocity': [],
            'spread_momentum': [],
            'return_volatility': []
        }

        for cycle in self.cycles_database:
            all_metrics['price_velocity'].append(cycle.early_price_velocity)
            all_metrics['spread_momentum'].append(cycle.early_spread_momentum)
            all_metrics['return_volatility'].append(cycle.early_return_volatility)

        # Calculate mean and std for normalization
        metric_stats = {}
        for metric, values in all_metrics.items():
            values = np.array(values)
            metric_stats[metric] = {
                'mean': np.nanmean(values),
                'std': np.nanstd(values)
            }

        # Normalise current metrics
        normalised_current = {}
        for metric in all_metrics.keys():
            if metric_stats[metric]['std'] > 0:
                normalised_current[metric] = ((current_metrics[metric] - metric_stats[metric]['mean'])
                                              / metric_stats[metric]['std'])
            else:
                normalised_current[metric] = 0

        print(f"Normalised current metrics: {normalised_current}")

        similarities = []
        for cycle in self.cycles_database:
            # Normalise cycle metrics
            cycle_metrics = {}
            for metric in all_metrics.keys():
                if metric_stats[metric]['std'] > 0:
                    cycle_metrics[metric] = ((getattr(cycle, f'early_{metric}') - metric_stats[metric]['mean'])
                                             / metric_stats[metric]['std'])
                else:
                    cycle_metrics[metric] = 0

            # Calculate similarity using normalised Euclidean distance
            distance = sum((normalised_current[k] - cycle_metrics[k]) ** 2
                           for k in all_metrics.keys()) ** 0.5

            similarities.append((cycle, 1 / (1 + distance)))

        # Sort by similarity and return top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [cycle for cycle, _ in similarities[:n_similar]]

    def _calculate_weighted_predictions(self,
                                        similar_cycles: [UnderOverCycle],
                                        weights: np.ndarray) -> CyclePrediction:
        """
        Calculate weighted predictions from similar cycles
        """
        if not similar_cycles:
            return None

        predictions = {
            'time_to_trough': [],
            'time_trough_to_under_end': [],
            'time_under_end_to_peak': [],
            'trough_return': [],
            'trough_to_under_end_return': [],
            'under_end_to_peak_return': [],
            'total_return': []
        }

        for cycle in similar_cycles:
            print(f"Individual cycle return: {cycle.trough_to_peak_return}")

            predictions['time_to_trough'].append(cycle.under_trough_idx - cycle.under_start_idx)
            predictions['time_trough_to_under_end'].append(cycle.under_end_idx - cycle.under_trough_idx)
            predictions['time_under_end_to_peak'].append(cycle.over_peak_idx - cycle.over_start_idx)
            predictions['trough_return'].append(cycle.trough_return)
            predictions['trough_to_under_end_return'].append(cycle.trough_to_under_end_return)
            predictions['under_end_to_peak_return'].append(cycle.peak_return)
            predictions['total_return'].append(cycle.trough_to_peak_return)

        # Calculate weighted averages
        means = {k: np.average(v, weights=weights) for k, v in predictions.items()}
        print(f"Calculated means: {means}")

        return CyclePrediction(
            expected_remaining_time_to_trough=int(means['time_to_trough']),
            expected_time_from_trough_to_under_end=int(means['time_trough_to_under_end']),
            expected_time_from_under_end_to_peak=int(means['time_under_end_to_peak']),
            expected_trough_return=means['trough_return'],
            expected_trough_to_under_end_return=means['trough_to_under_end_return'],
            expected_under_end_to_peak_return=means['under_end_to_peak_return'],
            expected_total_return=means['total_return'],
            confidence_score=np.max(weights)  # Use highest weight as confidence
        )

    def identify_cycles(self, series_a: pd.Series, series_b: pd.Series) -> [UnderOverCycle]:
        """
        Identify all under-over cycles in the data
        """
        spread = series_a - series_b
        cycles = []

        # Find crossover points
        spread_sign = np.sign(spread)
        crossovers = np.where(np.diff(spread_sign))[0]

        if len(crossovers) < 2:
            return cycles

        # Look for under-over pairs
        for i in range(len(crossovers) - 1):
            # Check if this is the start of an under period (positive to negative)
            if spread_sign.iloc[crossovers[i] + 1] < 0:
                under_start = crossovers[i]
                under_end = crossovers[i + 1]

                # Need enough points for early indicators
                if under_end - under_start < self.early_period_window:
                    continue

                under_slice = series_a[under_start:under_end]
                trough_idx = under_start + under_slice.values.argmin()

                # Look for subsequent over period
                if i + 2 < len(crossovers):
                    over_end = crossovers[i + 2]
                    over_slice = series_a[under_end:over_end]
                    peak_idx = under_end + over_slice.values.argmax()

                    cycle = UnderOverCycle(
                        under_start_idx=under_start,
                        under_trough_idx=trough_idx,
                        under_end_idx=under_end,
                        under_duration=under_end - under_start,
                        trough_return=(series_a.iloc[trough_idx] / series_a.iloc[under_start]) - 1,
                        trough_to_under_end_return=(series_a.iloc[under_end] / series_a.iloc[trough_idx]) - 1,

                        over_start_idx=under_end,
                        over_peak_idx=peak_idx,
                        over_end_idx=over_end,
                        over_duration=over_end - under_end,
                        peak_return=(series_a.iloc[peak_idx] / series_a.iloc[under_end]) - 1,

                        trough_to_peak_return=(series_a.iloc[peak_idx] / series_a.iloc[trough_idx]) - 1,
                        trough_to_peak_duration=peak_idx - trough_idx,

                        early_price_velocity=self._calculate_price_velocity(
                            series_a.iloc[under_start:under_start + self.early_period_window]
                        ),
                        early_spread_momentum=self._calculate_spread_momentum(
                            series_a.iloc[under_start:under_start + self.early_period_window],
                            series_b.iloc[under_start:under_start + self.early_period_window]
                        ),
                        early_return_volatility=self._calculate_return_volatility(
                            series_a.iloc[under_start:under_start + self.early_period_window]
                        ),
                        early_period_duration=min(self.early_period_window, under_end - under_start)
                    )

                    cycles.append(cycle)

        return cycles

    def predict_cycle(self,
                      mid_data: pd.Series,
                      ref_data: pd.Series,
                      duration: int,
                      n_similar: int = 5) -> CyclePrediction:
        current_metrics = {
            'mid_return': self._calculate_return(mid_data),
            'spread_momentum': self._calculate_spread_momentum(mid_data, ref_data),
            'mid_volatility': self._calculate_volatility(mid_data),
        }

        similar_cycles = self._find_similar_cycles(current_metrics, n_similar)
        print(f"Found {len(similar_cycles)} similar cycles")

        if not similar_cycles:
            return None

        # Calculate average metrics from similar cycles
        predictions = {
            'time_to_trough': [],
            'time_trough_to_under_end': [],
            'time_under_end_to_peak': [],
            'trough_return': [],
            'trough_to_under_end_return': [],
            'under_end_to_peak_return': [],
            'total_return': []
        }

        for i, cycle in enumerate(similar_cycles):
            print(f"Similar cycle {i}: trough_to_peak_return = {cycle.trough_to_peak_return}")
            predictions['time_to_trough'].append(cycle.under_trough_idx - cycle.under_start_idx)
            predictions['time_trough_to_under_end'].append(cycle.under_end_idx - cycle.under_trough_idx)
            predictions['time_under_end_to_peak'].append(cycle.over_peak_idx - cycle.over_start_idx)
            predictions['trough_return'].append(cycle.trough_return)
            predictions['trough_to_under_end_return'].append(cycle.trough_to_under_end_return)
            predictions['under_end_to_peak_return'].append(cycle.peak_return)
            predictions['total_return'].append(cycle.trough_to_peak_return)

        # Calculate means and standard deviations for confidence scoring
        means = {k: np.mean(v) for k, v in predictions.items()}
        stds = {k: np.std(v) for k, v in predictions.items()}

        # Calculate confidence score based on consistency of predictions
        confidence_score = np.mean([1 / (1 + std) for std in stds.values()])

        return CyclePrediction(
            expected_remaining_time_to_trough=int(means['time_to_trough']),
            expected_time_from_trough_to_under_end=int(means['time_trough_to_under_end']),
            expected_time_from_under_end_to_peak=int(means['time_under_end_to_peak']),
            expected_trough_return=means['trough_return'],
            expected_trough_to_under_end_return=means['trough_to_under_end_return'],
            expected_under_end_to_peak_return=means['under_end_to_peak_return'],
            expected_total_return=means['total_return'],
            confidence_score=confidence_score
        )

    def create_cycle_summary(self, output_path: Path) -> pd.DataFrame:
        """
        Create summary statistics for all cycles
        """
        if not self.cycles_database:
            print("Warning: No cycles in database to summarize")
            return pd.DataFrame()

        summary_data = []
        for cycle in self.cycles_database:
            # Extract cycle metrics
            summary_data.append({
                # Duration metrics
                'Under Period Duration': cycle.under_duration,
                'Time to Trough': cycle.under_trough_idx - cycle.under_start_idx,
                'Recovery Duration': cycle.under_end_idx - cycle.under_trough_idx,
                'Over Period Duration': cycle.over_duration,
                'Full Cycle Duration': cycle.under_duration + cycle.over_duration,
                'Trough to Peak Duration': cycle.trough_to_peak_duration,

                # Return metrics
                'Drawdown to Trough (%)': cycle.trough_return * 100,
                'Recovery Return (%)': cycle.trough_to_under_end_return * 100,
                'Over Period Return (%)': cycle.peak_return * 100,
                'Total Trough to Peak Return (%)': cycle.trough_to_peak_return * 100,

                # Early indicators
                'Early Price Velocity': cycle.early_price_velocity,
                'Early Spread Momentum': cycle.early_spread_momentum,
                'Early Return Volatility': cycle.early_return_volatility,
                'Early Period Duration': cycle.early_period_duration
            })

        df = pd.DataFrame(summary_data)

        # Add some additional statistical measures
        stats_df = pd.DataFrame({
            'Mean': df.mean(),
            'Median': df.median(),
            'Std': df.std(),
            'Min': df.min(),
            'Max': df.max(),
            'Count': len(df)
        }).T

        # Save both detailed and summary statistics
        df.to_csv(output_path / 'cycle_details.csv', index=False)
        stats_df.to_csv(output_path / 'cycle_statistics.csv')

        # Print key metrics
        print("\nKey Cycle Statistics:")
        print(f"Number of cycles: {len(df)}")
        print("\nAverage Durations (data points):")
        print(f"Full cycle: {df['Full Cycle Duration'].mean():.1f}")
        print(f"Time to trough: {df['Time to Trough'].mean():.1f}")
        print(f"Recovery time: {df['Recovery Duration'].mean():.1f}")

        print("\nAverage Returns (%):")
        print(f"Drawdown to trough: {df['Drawdown to Trough (%)'].mean():.2f}%")
        print(f"Recovery return: {df['Recovery Return (%)'].mean():.2f}%")
        print(f"Total return: {df['Total Trough to Peak Return (%)'].mean():.2f}%")

        return df

    def plot_cycle_characteristics(self, output_dir: Path):
        """
        Create visualizations of cycle characteristics
        """
        plt.figure(figsize=(15, 10))

        # Duration vs Returns scatter plot
        plt.subplot(2, 2, 1)
        durations = [c.under_duration + c.over_duration for c in self.cycles_database]
        returns = [c.trough_to_peak_return * 100 for c in self.cycles_database]
        plt.scatter(durations, returns, alpha=0.6)
        plt.xlabel('Cycle Duration')
        plt.ylabel('Total Return (%)')
        plt.title('Cycle Duration vs Total Return')

        # Return components boxplot
        plt.subplot(2, 2, 2)
        return_data = {
            'Trough': [c.trough_return * 100 for c in self.cycles_database],
            'Recovery': [c.trough_to_under_end_return * 100 for c in self.cycles_database],
            'Peak': [c.peak_return * 100 for c in self.cycles_database]
        }
        plt.boxplot(return_data.values())
        plt.xticks(range(1, 4), return_data.keys())
        plt.ylabel('Return (%)')
        plt.title('Return Components Distribution')

        # Save plot
        plt.tight_layout()
        plt.savefig(output_dir / 'cycle_characteristics.png')
        plt.close()

    def update_database(self, new_cycle: UnderOverCycle):
        """
        Add new completed cycle to database
        """
        # Calculate shift level for new cycle
        if self.trading_reference is not None:
            spread = new_cycle.series_a - self.trading_reference
            typical_spread = np.median(abs(spread))
            new_cycle.shift_level = spread.mean() / typical_spread

        self.cycles_database.append(new_cycle)

    def create_cycle_from_periods(self,
                                  periods: [PeriodState],
                                  price_series: pd.Series,
                                  reference_series: pd.Series) -> Optional[UnderOverCycle]:
        """
        Create a cycle from sequence of periods with proper early indicators
        """
        if len(periods) != 3:
            return None

        under_period, over_period, next_under = periods

        # Validate period sequence
        if not (under_period.type == 'under' and
                over_period.type == 'over' and
                next_under.type == 'under'):
            return None

        # Get early period data (first 20 points of under period)
        early_start = under_period.start_idx
        early_end = min(early_start + self.early_period_window, under_period.end_idx)

        early_prices = price_series[early_start:early_end]
        early_refs = reference_series[early_start:early_end]

        # Calculate returns
        trough_return = (under_period.extreme_price / under_period.start_price) - 1
        trough_to_under_end_return = (under_period.end_price / under_period.extreme_price) - 1
        peak_return = (over_period.extreme_price / over_period.start_price) - 1
        trough_to_peak_return = (over_period.extreme_price / under_period.extreme_price) - 1

        return UnderOverCycle(
            # Under period characteristics
            under_start_idx=under_period.start_idx,
            under_trough_idx=under_period.extreme_idx,
            under_end_idx=under_period.end_idx,
            under_duration=under_period.duration,
            trough_return=trough_return,
            trough_to_under_end_return=trough_to_under_end_return,

            # Over period characteristics
            over_start_idx=over_period.start_idx,
            over_peak_idx=over_period.extreme_idx,
            over_end_idx=over_period.end_idx,
            over_duration=over_period.duration,
            peak_return=peak_return,

            # Full cycle metrics
            trough_to_peak_return=trough_to_peak_return,
            trough_to_peak_duration=over_period.extreme_idx - under_period.extreme_idx,

            # Early indicators using actual data
            early_price_velocity=self._calculate_price_velocity(early_prices),
            early_spread_momentum=self._calculate_spread_momentum(early_prices, early_refs),
            early_return_volatility=self._calculate_return_volatility(early_prices),
            early_period_duration=len(early_prices)
        )

class PeriodTracker:
    def __init__(self, tolerance: int = 5):
        self.completed_periods: [PeriodState] = []
        self.current_period: Optional[PeriodState] = None
        self.tolerance = tolerance  # number of points to confirm period change
        self.recent_spreads = []  # keep last few spreads for tolerance check

    def should_change_period(self, new_spread: float) -> bool:
        """
        Check if we should change period based on spread history
        """
        self.recent_spreads.append(new_spread)
        if len(self.recent_spreads) > self.tolerance:
            self.recent_spreads.pop(0)

        if len(self.recent_spreads) < 2:
            return False

        # Check if majority of recent spreads confirm the change
        if self.current_period and self.current_period.type == 'under':
            # Currently in under period, check if we should switch to over
            positive_count = sum(1 for s in self.recent_spreads[-2:] if s > 0)
            return positive_count == 2
        else:
            # Currently in over period or no period, check if we should switch to under
            negative_count = sum(1 for s in self.recent_spreads[-2:] if s < 0)
            return negative_count == 2

    def start_new_period(self, idx: int, price: float, spread: float) -> None:
        """
        Start a new period
        """
        # Complete current period if exists
        if self.current_period:
            self.current_period.complete(idx, price, spread)
            self.completed_periods.append(self.current_period)

        # Determine new period type
        new_type = 'under' if spread < 0 else 'over'

        # Create new period
        self.current_period = PeriodState(
            type=new_type,
            start_idx=idx,
            start_price=price,
            start_spread=spread
        )
        print(f"Started new {new_type} period at index {idx}")

    def update(self, idx: int, price: float, spread: float) -> bool:
        """
        Update period tracking
        """
        # Initialise if no current period
        if self.current_period is None:
            self.start_new_period(idx, price, spread)
            return True

        # Check if we should change period
        if self.should_change_period(spread):
            self.start_new_period(idx, price, spread)
            return True

        # Update current period
        self.current_period.update(idx, price, spread)
        return False

    def get_last_n_periods(self, n: int) -> [PeriodState]:
        """
        Get the last n completed periods plus current period if it exists
        """
        result = self.completed_periods[-n:] if n < len(self.completed_periods) else self.completed_periods[:]
        if self.current_period:
            result.append(self.current_period)
        return result

    def is_cycle_complete(self) -> bool:
        """
        Check if we just completed a cycle (under->over)
        """
        if len(self.completed_periods) < 2 or not self.current_period:
            return False

        penultimate = self.completed_periods[-2]
        last = self.completed_periods[-1]
        return (penultimate.type == 'under' and
                last.type == 'over' and
                self.current_period.type == 'under')

    def __str__(self) -> str:
        status = f"PeriodTracker: {len(self.completed_periods)} completed periods\n"
        if self.current_period:
            status += f"Current: {self.current_period}"
        return status