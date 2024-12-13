import pandas as pd
import numpy as np

from concepts import OrderType, OrderSide, TradeReason
from data_processing import OrderBookFeed
from execution import TradeManager, PositionManager, OrderTypeDecision, PositionUpdate, TradeCostCalculator
from configs import MarketConfig, TradingConfig, DataConfig
from time_manager import TimeEngine
from reporting import CyclePerformanceTracker
from period_cycle import (Cycle, CycleTracker, identify_all_periods, construct_cycles, start_of_last_under_period,
                          CyclePrediction)


class TradingSimulator:
    def __init__(self,
                 market_config: MarketConfig,
                 trading_config: TradingConfig,
                 data_config: DataConfig,
                 ):
        # Configs
        self.market_config = market_config
        self.trading_config = trading_config
        self.data_config = data_config

        # Components
        self.database = {'cycles': [], 'periods': []}
        self.cycle_tracker = CycleTracker(self.data_config.MIN_CONSECUTIVE_POINTS)
        self.position_manager = PositionManager(trading_config, market_config, data_config)
        self.trade_manager = TradeManager(market_config, trading_config)
        self.performance_tracker = CyclePerformanceTracker(market_config)
        self.cost_calculator = TradeCostCalculator(market_config)

        # State tracking
        self.is_collecting_data = False
        self.cycle_start_idx = None
        self.data_points_collected = 0

    def build_cycle_database(self, mid_series: pd.Series, reference_series: pd.Series, idx_start_ref: int = 0) -> [Cycle]:
        """
        Build initial cycle database from training data using period and cycle identification
        """
        print("Building initial cycle database...")

        # First identify all periods in training data
        periods = identify_all_periods(mid_series, reference_series, self.data_config.MIN_CONSECUTIVE_POINTS,
                                       idx_start_ref)

        # Construct cycles from periods
        cycles = construct_cycles(periods)

        if periods[-1].type == 'under':
            periods.pop()
            cycles.pop()
        else:
            periods = periods[:-2]
            cycles.pop()
        idx_trading_start = periods[-1].end_idx + 1
        # Store cycles in analyzer database
        self.database = {'cycles': cycles, 'periods': periods}

        print(f"Found {len(cycles)} cycles in training data")

        # Create and print summary
        cycle_summary = pd.DataFrame([{
            'Duration': cycle.duration,
            'Start to Trough Time': cycle.trough_idx - cycle.start_idx,
            'Trough to Peak Time': cycle.peak_idx - cycle.trough_idx,
            'Peak to End Time': cycle.end_idx - cycle.peak_idx,
            'Start to Trough Return': cycle.start_to_trough_return,
            'Trough to Peak Return': cycle.trough_to_peak_return,
            'Peak to End Return': cycle.peak_to_end_return,
            'Total Return': cycle.total_return
        } for cycle in cycles])

        print("\nCycle Summary Statistics:")
        print(cycle_summary.describe())
        return idx_trading_start

    def _calculate_return(self, series: pd.Series) -> float:
        """Calculate momentum of price movements"""
        return float(0 if abs(series.iloc[0]) < 1e-10 else series.iloc[-1] / series.iloc[0] - 1)

    def _calculate_spread_change(self, series_a: pd.Series, series_b: pd.Series) -> float:
        """Calculate momentum of the spread"""
        spread = series_a - series_b
        return spread.iloc[-1] - spread.iloc[0]
    def _calculate_volatility(self, series: pd.Series, window: int = 20) -> float:
        """Calculate rolling volatility of returns"""
        returns = series.pct_change().fillna(0)
        if len(returns) < window:
            return 0.0
        vol = returns.rolling(window=window).std().iloc[-1]
        return 0.0 if np.isnan(vol) else vol

    def _find_similar_cycles(self, current_metrics: dict[str, float], n_similar: int = 20) -> [Cycle]:
        """Find similar cycles using normalised metrics"""
        print(f"Finding similar cycles from database of {len(self.database['cycles'])} cycles")
        print(f"Current metrics before normalization: {current_metrics}")

        # Collect all metric values for normalization
        all_metrics = {
            'mid_return': [],
            'spread_change': [],
            'mid_volatility': [],

        }

        for cycle in self.database['cycles'][:-1]:
            all_metrics['mid_return'].append(cycle.early_mid_return)
            all_metrics['spread_change'].append(cycle.early_spread_change)
            all_metrics['mid_volatility'].append(cycle.early_mid_volatility)

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
        for cycle in self.database['cycles'][:-1]:
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
        return [cycle for cycle, _ in similarities[:min(n_similar, len(similarities))]]

    def predict_cycle(self,
                      mid_data: pd.Series,
                      ref_data: pd.Series,
                      n_similar: int = 20) -> CyclePrediction:
        current_metrics = {
            'mid_return': self._calculate_return(mid_data),
            'spread_change': self._calculate_spread_change(mid_data, ref_data),
            'mid_volatility': self._calculate_volatility(mid_data),
        }

        similar_cycles = self._find_similar_cycles(current_metrics, n_similar)
        print(f"Found {len(similar_cycles)} similar cycles")

        if not similar_cycles:
            return None

        # Calculate average metrics from similar cycles
        predictions = {
            'time_to_trough': [],
            'time_trough_to_peak': [],
            'trough_to_peak_return': [],
        }

        for i, cycle in enumerate(similar_cycles):
            print(f"Similar cycle {i}: trough_to_peak_return = {cycle.trough_to_peak_return}")
            predictions['time_to_trough'].append(cycle.trough_idx - cycle.start_idx)
            predictions['time_trough_to_peak'].append(cycle.peak_idx - cycle.trough_idx)
            predictions['trough_to_peak_return'].append(cycle.trough_to_peak_return)


        # Calculate means and standard deviations for confidence scoring
        means = {k: np.mean(v) for k, v in predictions.items()}
        stds = {k: np.std(v) for k, v in predictions.items()}

        # Calculate confidence score based on consistency of predictions
        confidence_score = np.mean([1 / (1 + std) for std in stds.values()])

        return CyclePrediction(
            time_to_trough=int(max(means['time_to_trough']-self.data_config.MIN_POINTS_IN_PERIOD, 0)),
            time_trough_to_peak=int(means['time_trough_to_peak']),
            trough_to_peak_return=float(means['trough_to_peak_return']),
            confidence_score=float(confidence_score)
        )

    def run_sim(self, mid_series: pd.Series, reference_series: pd.Series, training_cutoff: int) -> dict:
        time_engine = TimeEngine(mid_series[training_cutoff:], self.market_config.MAKER_FILL_SECONDS_STEP)
        order_book_feed = OrderBookFeed(self.market_config)

        print("Starting simulation...")

        # Create index mapping for timestamps
        timestamp_to_idx = {ts: idx for idx, ts in enumerate(mid_series.index)}

        for time_slice in time_engine.generate_time_slices():
            current_idx = timestamp_to_idx.get(time_slice.timestamp)
            if current_idx is None:
                print('Checking MAKER fill')
            else:
                print(current_idx)
            # process maker order fills even when there is no new market data
            if time_slice.data_point is None:
                trade = self.trade_manager.process_time_slice(time_slice)
                if trade:
                    print(f"Maker fill executed: size={trade.size:.4f}, price={trade.price:.6f}")
                    self.performance_tracker.record_trade(trade)
                    self.position_manager.update_position(trade)
                continue

            # Update order book
            order_book = order_book_feed.update(time_slice.data_point)

            # Track peak price if position is active
            if self.position_manager.state.is_active:
                self.performance_tracker.update_peak_price(order_book.mid)

            # get start index of last complete 'under' period
            last_under_start_idx, complete_last_under_in_db = start_of_last_under_period(self.database['periods'])

            # get cycles from current window
            cycles, periods = self.cycle_tracker.update(
                mid_series=mid_series[last_under_start_idx:current_idx + 1],
                reference_series=reference_series[last_under_start_idx:current_idx + 1],
                idx_start_ref=last_under_start_idx
            )
            if complete_last_under_in_db:
                # periods consists of 'under', 'over', 'under' or 'under', 'over'
                self.database['periods'].pop()
                self.database['periods'].extend(periods[1:])
                self.database['cycles'].pop()
                self.database['cycles'].extend(cycles)
            else:
                # periods consists of 'under', 'over' or 'under'
                self.database['periods'].pop()
                self.database['periods'].extend(periods)
                self.database['cycles'].pop()
                self.database['cycles'].extend(cycles)

            # Trading logic using current cycle
            current_cycle = cycles[-1]  # Latest cycle for trading decisions

            # Handle position management only if we don't have an active position
            if not self.position_manager.state.is_active:
                # Only generate predictions if cycle duration is sufficient
                if current_cycle.duration == self.data_config.MIN_POINTS_IN_PERIOD:
                    prediction = self.predict_cycle(
                        mid_data=mid_series[current_cycle.start_idx:current_idx + 1],
                        ref_data=reference_series[current_cycle.start_idx:current_idx + 1],
                    )

                    if prediction:
                        position_size, order_decision = self.position_manager.calculate_position_size_order_type(
                            prediction,
                            current_idx
                        )
                        self.position_manager.state.expected_peak_index = (
                                current_idx + prediction.time_to_trough + prediction.time_trough_to_peak
                        )
                        # Start tracking new cycle when entering position
                        self.performance_tracker.start_new_cycle(prediction, order_book.mid)

                        if position_size > 0:
                            position_update = PositionUpdate(
                                size=position_size,
                                side=OrderSide.BUY,
                                timestamp=time_slice.timestamp
                            )

                            trade = self.trade_manager.execute_trade(
                                position_update=position_update,
                                order_book=order_book,
                                order_type_decision=order_decision,
                                current_idx=current_idx,
                                entry_reason=TradeReason.CYCLE_ENTRY
                            )

                            if trade:
                                print(f"Entry trade executed: size={trade.size:.4f}, price={trade.price:.6f}")
                                self.performance_tracker.record_trade(trade)
                                self.position_manager.update_position(trade)

                            if order_decision.order_type == OrderType.MAKER:
                                time_engine.start_fill_checks(time_slice.timestamp)

            # Handle exit conditions if we have an active position
            if self.position_manager.state.is_active:
                # Exit if new cycle started OR peak reached OR stop loss hit
                exit_needed = (
                        len(cycles) == 2 or
                        current_idx >= self.position_manager.state.expected_peak_index
                        # or (
                        #         order_book.bid / self.position_manager.state.entry_price - 1) <=
                        # -self.trading_config.STOP_LOSS_PCT
                )

                if exit_needed:
                    if len(cycles) == 2:
                        exit_reason = TradeReason.CYCLE_NEW_UNDER
                    elif current_idx >= self.position_manager.state.expected_peak_index:
                        exit_reason = TradeReason.CYCLE_PEAK_EXIT
                    else:
                        exit_reason = TradeReason.CYCLE_STOP_LOSS

                    position_update = PositionUpdate(
                        size=self.position_manager.state.size,
                        side=OrderSide.SELL,
                        timestamp=time_slice.timestamp
                    )

                    costs = self.cost_calculator.calculate_taker_costs(position_update.size, position_update.side)
                    order_decision = OrderTypeDecision(
                        order_type=OrderType.TAKER,
                        price_impact=costs.price_impact,
                        fill_time=costs.fill_time,
                        modified_size=position_update.size,
                        fee=costs.fee_component,
                        slippage=costs.slippage_component
                    )

                    trade = self.trade_manager.execute_trade(
                        position_update=position_update,
                        order_book=order_book,
                        order_type_decision=order_decision,
                        current_idx=current_idx,
                        exit_reason=exit_reason
                    )

                    if trade:
                        print(
                            f"Exit trade executed: size={trade.size:.4f}, price={trade.price:.6f}, reason={exit_reason.name}")
                        self.performance_tracker.record_trade(trade)
                        self.position_manager.update_position(trade)
                        self.performance_tracker.complete_cycle(
                            current_cycle=current_cycle,
                            exit_reason=exit_reason
                        )
                        time_engine.stop_fill_checks()

        print("\nSimulation completed:")
        print(f"Final cycles in database: {len(self.database['cycles'])}")

        return {
            'performance_report': self.performance_tracker.generate_performance_report(),
            'trades': self.trade_manager.get_trades(),
            'cycle_performances': self.performance_tracker.performances,
            'final_cycle_count': len(self.database['cycles'])
        }
