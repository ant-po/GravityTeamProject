from dataclasses import dataclass
from typing import Optional, Iterator
import pandas as pd


@dataclass
class TimeSlice:
    timestamp: pd.Timestamp  # Single timestamp representing the point in time
    data_point: Optional[float]  # None for fill checks, Series for market data


class TimeEngine:
    def __init__(self, data: pd.Series, maker_fill_seconds_step: float):
        self.data = data
        self.maker_fill_seconds_step = maker_fill_seconds_step
        self.next_fill_time = None  # Only set when maker order is active

    def generate_time_slices(self) -> Iterator[TimeSlice]:
        """
        Generate time slices for market data and fill checks
        """
        data_index = self.data.index.tolist()

        for i, timestamp in enumerate(data_index[:-1]):
            # always yield the data point
            yield TimeSlice(timestamp, float(self.data.iloc[i]))

            # if we have an active maker order, generate fill checks until next data point
            if self.next_fill_time is not None:
                next_data_time = data_index[i + 1]

                while self.next_fill_time < next_data_time:
                    yield TimeSlice(self.next_fill_time, None)
                    self.next_fill_time += pd.Timedelta(seconds=self.maker_fill_seconds_step)

        # yield the last market data point
        yield TimeSlice(data_index[-1], float(self.data.iloc[-1]))

    def start_fill_checks(self, current_time: pd.Timestamp):
        """
        Need to call this when a maker order is placed to start fill checks
        """
        self.next_fill_time = current_time + pd.Timedelta(seconds=self.maker_fill_seconds_step)

    def stop_fill_checks(self):
        """
        Need to call this when maker order is completed/cancelled
        """
        self.next_fill_time = None
