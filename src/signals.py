from turtledemo.sorting_animate import start_isort

import numpy as np
import pandas as pd

from typing import Optional
from dataclasses import dataclass
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller

from utils import sigmoid
from concepts import Signal, TradeReason


class CorSignal:
    def __init__(self, window: int = 100):
        self.window = window

    def calculate_signal(self, data: pd.Series, reference: pd.Series, timestamp: pd.Timestamp):
        rolling_correlation = data.rolling(window=self.window, min_periods=100).corr(reference).fillna(0)
        if rolling_correlation.iloc[-1] > 0.5:
            target = 1
        elif rolling_correlation.iloc[-1] < -0.5:
            target = 0
        else:
            target = 0.5
        return Signal(target=target, conviction=0.5, timestamp=timestamp)


class ZScoreSignal:
    def __init__(self,
                 window: int = 100,
                 conviction_scalar: float = 2.0):
        self.window = window
        self.conviction_scalar = conviction_scalar

    def calculate_signal(self, spread: pd.Series, timestamp: pd.Timestamp) -> (Signal, float):
        # calculate rolling stats
        rolling_mean = spread.rolling(window=self.window, min_periods=1).mean()
        rolling_std = spread.rolling(window=self.window, min_periods=1).std()

        # current z-score
        z_score = (spread.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        if np.isnan(z_score):
            return Signal(target=0.0, conviction=0.0, timestamp=timestamp), 0
        target = 1.0 if z_score < 0 else 0.0
        conviction = sigmoid(abs(z_score) / self.conviction_scalar)
        return Signal(target=target, conviction=conviction, timestamp=timestamp), z_score


@dataclass
class SpreadCharacteristics:
    half_life: float  # Half-life of mean reversion
    mean_level: float  # Long-term mean level
    volatility: float  # Spread volatility
    seasonality: Optional[dict]  # Seasonal patterns if any
    structural_bias: float  # Persistent bias in spread
    correlation_decay: float  # How fast correlation decays


class EnhancedSpreadSignal:
    def __init__(
            self,
            estimation_window: int = 1000,
            signal_window: int = 100,
            seasonal_window: int = 24,  # For hourly seasonality
            min_half_life: float = 60.0,  # Minimum half-life in seconds
            max_half_life: float = 3600.0,  # Maximum half-life in seconds
            zscore_threshold: float = 2.0
    ):
        self.estimation_window = estimation_window
        self.signal_window = signal_window
        self.seasonal_window = seasonal_window
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.zscore_threshold = zscore_threshold
        self.spread_characteristics = None

    @staticmethod
    def calculate_half_life(spread: pd.Series) -> float:
        """
        Calculate the half-life of mean reversion using OLS
        """
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        spread_lag = spread_lag.dropna()
        spread_diff = spread_diff.dropna()

        model = OLS(spread_diff, spread_lag).fit()
        half_life = -np.log(2) / model.params[0]
        return half_life

    @staticmethod
    def detect_seasonality(spread: pd.Series) -> dict:
        """
        Detect seasonal patterns in the spread
        """
        # convert timestamp to hour
        hours = pd.Series(spread.index.hour, index=spread.index)

        # calculate mean spread by hour
        hourly_means = spread.groupby(hours).mean()
        hourly_stds = spread.groupby(hours).std()

        # test for significance of seasonal patterns
        f_stat, p_value = stats.f_oneway(*[
            spread[hours == hour] for hour in range(24)
        ])

        return {
            'hourly_means': hourly_means,
            'hourly_stds': hourly_stds,
            'is_significant': p_value < 0.05,
            'f_stat': f_stat,
            'p_value': p_value
        }

    @staticmethod
    def calculate_structural_bias(spread: pd.Series) -> float:
        """
        Calculate persistent bias in the spread
        """
        # use Augmented Dickey-Fuller test to check stationarity
        adf_result = adfuller(spread)

        if adf_result > 0.05:  # Non-stationary
            return spread.mean()
        return 0.0

    def analyze_spread_characteristics(self, spread: pd.Series) -> SpreadCharacteristics:
        """Analyze key characteristics of the spread"""
        # calculate half-life of mean reversion
        half_life = self.calculate_half_life(spread)
        half_life = np.clip(half_life, self.min_half_life, self.max_half_life)

        # calculate basic statistics
        mean_level = spread.mean()
        volatility = spread.std()

        # detect seasonality
        seasonality = self.detect_seasonality(spread)

        # calculate structural bias
        structural_bias = self.calculate_structural_bias(spread)

        # calculate correlation decay
        lags = range(1, 11)
        autocorr = [spread.autocorr(lag=lag) for lag in lags]
        correlation_decay = -np.polyfit(lags, np.log(np.abs(autocorr)), deg=1)[0]

        return SpreadCharacteristics(
            half_life=half_life,
            mean_level=mean_level,
            volatility=volatility,
            seasonality=seasonality,
            structural_bias=structural_bias,
            correlation_decay=correlation_decay
        )

    def calculate_signal(self, spread: pd.Series, timestamp: pd.Timestamp) -> Signal:
        """
        Generate trading signal based on spread characteristics
        """
        # update spread characteristics periodically
        if self.spread_characteristics is None or len(spread) % self.estimation_window == 0:
            self.spread_characteristics = self.analyze_spread_characteristics(
                spread.iloc[-self.estimation_window:]
            )

        # get current spread properties
        current_spread = spread.iloc[-1]
        current_hour = timestamp.hour

        # calculate seasonal adjustment if significant
        seasonal_adj = 0.0
        if (self.spread_characteristics.seasonality and
                self.spread_characteristics.seasonality['is_significant']):
            seasonal_adj = (
                    self.spread_characteristics.seasonality['hourly_means'][current_hour] -
                    self.spread_characteristics.mean_level
            )

        # calculate adjusted spread level
        adjusted_spread = (
                current_spread -
                self.spread_characteristics.structural_bias -
                seasonal_adj
        )

        # calculate dynamic z-score
        recent_vol = spread.iloc[-self.signal_window:].std()
        zscore = (
                         adjusted_spread - self.spread_characteristics.mean_level
                 ) / recent_vol

        # calculate mean-reversion strength
        reversion_strength = np.exp(-np.log(2) / self.spread_characteristics.half_life)

        # calculate position target
        if abs(zscore) > self.zscore_threshold:
            # base position size on z-score magnitude and mean-reversion strength
            raw_target = -np.sign(zscore) * (
                    1 - np.exp(-abs(zscore - self.zscore_threshold))
            ) * reversion_strength

            # adjust for correlation decay
            target = raw_target * np.exp(
                -self.spread_characteristics.correlation_decay
            )

            # scale target based on volatility
            vol_scale = min(1.0, (
                    self.spread_characteristics.volatility / recent_vol
            ))
            target *= vol_scale

            # calculate conviction based on signal strength
            conviction = min(1.0, abs(zscore) / self.zscore_threshold)

            # adjust conviction based on seasonality
            if (self.spread_characteristics.seasonality and
                    self.spread_characteristics.seasonality['is_significant']):
                seasonal_vol = self.spread_characteristics.seasonality['hourly_stds'][current_hour]
                seasonal_scale = recent_vol / seasonal_vol
                conviction *= min(1.0, seasonal_scale)
        else:
            target = 0.0
            conviction = 0.0

        return Signal(
            target=np.clip(target, -1.0, 1.0),
            conviction=conviction,
            timestamp=timestamp
        )


@dataclass
class TrendMetrics:
    trend_direction: float  # 1 for up, -1 for down, 0 for neutral
    trend_strength: float  # 0 to 1
    momentum_score: float  # normalised momentum indicator
    volatility: float  # recent price volatility


@dataclass
class SpreadMetrics:
    z_score: float  # standardized spread
    mean_spread: float  # average spread over window
    spread_volatility: float  # spread volatility
    is_extreme: bool  # if spread is at extreme levels


class HybridSignal:
    def __init__(self, window: int = 200):
        """
        Simple signal combining trend and spread
        """
        self.window = window
        self.last_signal_time = None
        self.min_bars_between_trades = 50

    def calculate_signal(self, price_series: pd.Series, spread: pd.Series, timestamp: pd.Timestamp) -> \
            (Signal, float, TradeReason):
        """
        Generate trading signal based on:
        1. Trend of executable rate (moving average)
        2. Spread deviation from mean (z-score)
        """
        if len(spread) < self.window:
            return Signal(target=0.0, conviction=1.0, timestamp=timestamp), 0.0, None

            # enforce minimum time between trades
        if self.last_signal_time is not None:
            if len(price_series.loc[self.last_signal_time:timestamp]) < self.min_bars_between_trades:
                return Signal(target=0.0, conviction=1.0, timestamp=timestamp), 0.0, None

        # calculate trend (positive if price above MA, negative if below)
        price_ma = price_series.rolling(window=self.window).mean().iloc[-1]
        trend = 1 if price_series.iloc[-1] > price_ma else -1

        # calculate spread metrics
        mean_spread = spread.rolling(window=self.window).mean().iloc[-1]
        spread_std = spread.rolling(window=self.window).std().iloc[-1]
        zscore = (spread.iloc[-1] - mean_spread) / spread_std if spread_std > 0 else 0

        target = 0.0
        conviction = 0.0
        reason = None

        # entry/exit conditions
        if trend > 0:  # Stronger spread deviation required
            target = 1
            conviction = 1
            reason = TradeReason.TREND_SPREAD_ENTRY
        elif trend < 0:
            target = 0.0
            conviction = 1
            reason = TradeReason.TREND_REVERSAL
        elif zscore > 2.0:
            target = 0.0
            conviction = 1.0
            reason = TradeReason.SPREAD_EXTREME
        self.last_signal_time = timestamp
        return Signal(target=target, conviction=conviction, timestamp=timestamp), zscore, reason
