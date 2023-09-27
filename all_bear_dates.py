import pandas as pd
from typing import Optional, Literal, Union
from scipy import signal
import numpy as np
import fredpy as fp
from matplotlib import pyplot as plt
import datetime
from dotenv import load_dotenv

class BearData:
    def __init__(self, series: pd.Series) -> None:
        """
        converts series index to 'datetime.date' and saves as attribute
        """
        if all(isinstance(i, pd.Timestamp) for i in series.index):
            series.index = [date.date() for date in series.index]
            self.prices = series
        else:
            self.prices = series
        return None

    def get_technical_bear_dates(self):
        """
        Find all technical bear market start (top) and end (end) dates.
        1. A technical start is a peak in price data that is followed by a
        decline of 20% or more.
        2. A technical end is the date at which prices have recovered by 20%
        or more from a trough following a technical start.
        3. There cannot be a technical start until the previous bear market
        has technically ended.
        """
        prices = self.prices
        peaks, _ = signal.find_peaks(prices, distance=5)
        troughs, _ = signal.find_peaks(prices * -1, distance=5)
        prominences = signal.peak_prominences(self.prices, peaks)
        bear_dates = {("date", "top"): [], ("date", "end"): []}
        end = prices.index[0]
        for peak, prominence in zip(peaks, prominences[2]):
            is_end = False
            if (
                end
                and (prices.index[peak] - end).days >= 0
                and prices.iloc[peak] / prices.iloc[prominence] >= 1.2
            ):
                top = peak
                bear_dates["date", "top"].append(prices.index[top])
                troughs_after_top = prices.index[troughs[troughs > top]]
                for trough in troughs_after_top:
                    retracement = prices[trough:][
                        prices[trough:] >= prices[trough] * 1.2
                    ].first_valid_index()
                    if (
                        retracement
                        and prices[trough:retracement].idxmin() == trough
                    ):
                        is_end = True
                        end = retracement
                        bear_dates["date", "end"].append(end)
                        break
                if not is_end:
                    bear_dates["date", "end"].append(None)

        return pd.DataFrame(bear_dates)

    def get_filtered_bear_dates(self):
        """
        Make the following adjustments to technical bear dates:
        1. if a new bear market starts within 6 months of the previous one ending
        and highest price between the the 2 bear markets is at the initial top
        then concatenate these bear markets.
        2. Having carried out 1), if there are any bear markets that last for less
        than 6 months, remove them.
        3. Add trough (bottom) and new high (new_high) dates to each bear market
        """
        prices = self.prices
        df = self.get_technical_bear_dates()
        df["info", "bear_traps"] = 0
        i = 1

        while i < len(df):
            if (df.iloc[i, 0] - df.iloc[i - 1, 1]).days <= 180 and prices[
                df.iloc[i - 1, 0] : df.iloc[i, 1]
            ].idxmax() == df.iloc[i - 1, 0]:
                df.iloc[i - 1, 1] = df.iloc[i, 1]
                df.iloc[i - 1, 2] += 1
                df.drop(index=i, inplace=True)
                df.reset_index(drop=True, inplace=True)
            else:
                i += 1

        for row in df.iterrows():
            if (
                row[1]["date", "end"]
                and (row[1]["date", "end"] - row[1]["date", "top"]).days <= 180
            ):
                df.drop(index=row[0], inplace=True)

        df.insert(
            1,
            ("date", "bottom"),
            [
                prices[top:end].idxmin()
                if end
                else prices[top : prices.index[-1]].idxmin()
                for top, end in zip(df["date", "top"], df["date", "end"])
            ],
        )
        df.insert(
            3,
            ("date", "new_high"),
            [
                prices[top:][prices[top:] > prices[top]].first_valid_index()
                for top in df["date", "top"]
            ],
        )
        df["info", "duration"] = [
            (end - top).days if end else None
            for end, top in zip(df["date", "end"], df["date", "top"])
        ]

        return df.reset_index(drop=True)

    def get_transformed_series(
        self,
        series: pd.Series,
        start_date: datetime.date,
        change=False,
        pct_change=False,
    ) -> pd.Series:
        """Given a start date and a series, slice the series from the closest 
        date to the start date and either return the series as is or in the 
        form of changes or percentage changes"""

        if (start_date - series.index[0].date()).days >= 0:
            closest_date = min(
                series.index, key=lambda date: abs(date.date() - start_date)
            )
            series = series[closest_date:]
            if change:
                series = round(series - series[0], 2)
            if pct_change:
                series = round((series - series[0]) / series[0] * 100, 2)
        else:
            return None
        return series

    def concat_fp_series(
        self,
        df: pd.DataFrame,
        series: pd.Series,
        title: str,
        change=False,
        pct_change=False,
    ):
        """Add values of a given series to the dataframe containing bear dates"""

        df_dates = df["date"]
        result = {(title, col): [] for col in df_dates.columns}
        for row in df_dates.iterrows():
            transformed_series = self.get_transformed_series(
                series, row[1]["top"], change, pct_change
            )
            for _, key in result.keys():
                if transformed_series is not None and row[1][key]:
                    closest_date = min(
                        transformed_series.index,
                        key=lambda date: abs(date.date() - row[1][key]),
                    )
                    result[(title, key)].append(
                        transformed_series[closest_date]
                    )
                else:
                    result[(title, key)].append(None)

        if change or pct_change:
            result.pop((title, "top"))

        return pd.concat([df, pd.DataFrame(result, index=df.index)], axis=1)

    def concat_multiple_fp_series(self, df: pd.DataFrame, *args: tuple):
        """Add multiple series to bear dates df"""
        for arg in args:
            df = self.concat_fp_series(df, *arg)
        return df

    def plot_fp_series(
        self,
        df: pd.DataFrame,
        series: pd.Series,
        change=False,
        pct_change=False,
    ) -> None:
        """plot a series for every bear market in bear dates df"""
        fig, ax = plt.subplots(figsize=(20, 10))
        for row in df["date"].iterrows():
            series_slice = self.get_transformed_series(
                series, row[1]["top"], change, pct_change
            )
            if series_slice is not None and row[1]["end"]:
                if row[0] != df.index[-1]:
                    ax.plot(
                        series_slice[: row[1]["end"]].reset_index(drop=True),
                        color="grey",
                        alpha=0.6,
                    )
                else:
                    ax.plot(
                        series_slice[: row[1]["end"]].reset_index(drop=True),
                        color="blue",
                    )
