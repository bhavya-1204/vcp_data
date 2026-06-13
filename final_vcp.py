import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import requests
from io import StringIO

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


class VCPAnalyzer:
    """Analyzes a stock's OHLC data for a VCP (Volatility Contraction Pattern).
    Logic is unchanged from the original implementation.
    """

    def __init__(self, ema_period=9, min_distance=3, max_distance=20):
        self.ema_period = ema_period
        self.min_distance = min_distance
        self.max_distance = max_distance

    def calculate_ema(self, df):
        if len(df) < self.ema_period:
            return None
        return df['Close'].ewm(span=self.ema_period, adjust=False).mean()

    def analyze(self, data):
        ema_result = self.calculate_ema(data)
        if ema_result is None:
            return False, None, None
        data['09_ema'] = ema_result.round(2)

        # Low main
        low = data['Low'].min().item()
        low_idx = data[np.isclose(data['Low'], low)].index[0]
        index_number = data.index.get_loc(low_idx)
        data_1 = data.loc[low_idx:]

        # Peak 1
        high_1 = data_1['High'].max().item()
        high_1_idx = data_1[np.isclose(data_1['High'], high_1)].index[0]
        index_number_1 = data.index.get_loc(high_1_idx)

        data_2 = data_1.loc[high_1_idx:]

        # Low 1
        low_1 = data_2['Low'].min().item()
        low_1_idx = data_2[np.isclose(data_2['Low'], low_1)].index[0]
        index_number_2 = data.index.get_loc(low_1_idx)
        data_3 = data_2.loc[low_1_idx:]

        # Peak 2
        high_2 = data_3['High'].max().item()
        high_2_idx = data_3[np.isclose(data_3['High'], high_2)].index[0]
        index_number_3 = data.index.get_loc(high_2_idx)

        high_2_close = data_3.loc[high_2_idx, 'Close'].item()
        high_2_09 = data['09_ema'].loc[high_2_idx].item()

        if high_2_close < high_2_09:
            return False, None, None
        if high_2 > high_1:
            return False, None, None

        data_4 = data.loc[high_2_idx:]

        # Low 2
        low_2 = data_4['Low'].min().item()
        if low_2 < low_1:
            return False, None, None

        low_2_idx = data_4[np.isclose(data_4['Low'], low_2)].index[0]
        index_number_4 = data.index.get_loc(low_2_idx)

        data_5 = data.loc[low_2_idx:]

        # Peak 3
        high_3 = data_5['High'].max().item()
        high_3_idx = data_5[np.isclose(data_5['High'], high_3)].index[0]
        index_number_5 = data.index.get_loc(high_3_idx)

        high_3_close = data_5.loc[high_3_idx, 'Close'].item()
        high_3_09 = data['09_ema'].loc[high_3_idx].item()

        if high_3_close < high_3_09:
            return False, None, None
        if high_3 > high_2:
            return False, None, None

        indices = [index_number, index_number_1, index_number_2,
                   index_number_3, index_number_4, index_number_5]
        if len(set(indices)) != len(indices):
            return False, None, None

        distances = [
            index_number_1 - index_number,
            index_number_2 - index_number_1,
            index_number_3 - index_number_2,
            index_number_4 - index_number_3,
            index_number_5 - index_number_4,
        ]

        if not all(self.min_distance <= d <= self.max_distance for d in distances):
            return False, None, None

        return True, low_idx.date(), high_3_idx.date()


class NSEStockFetcher:
    """Fetches NSE equity symbol list and downloads OHLC data via yfinance."""

    NSE_HOME_URL = "https://www.nseindia.com"
    NSE_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/"
        }

    def get_nse_symbols(self):
        self.session.get(self.NSE_HOME_URL, headers=self.headers)
        response = self.session.get(self.NSE_CSV_URL, headers=self.headers)
        symbol_name_csv = pd.read_csv(StringIO(response.text))
        return [symbol + '.NS' for symbol in symbol_name_csv['SYMBOL']]

    def download_data(self, ticker, period='5y', interval='1d'):
        return yf.download(ticker, period=period, interval=interval, progress=False).round(2)


class VCPScanner:
    """Scans NSE stocks for the VCP pattern (current snapshot) and returns matching results.
    Unchanged from before.
    """

    def __init__(self, min_price=100, min_volume=150000):
        self.min_price = min_price
        self.min_volume = min_volume
        self.fetcher = NSEStockFetcher()
        self.analyzer = VCPAnalyzer()

    def _passes_filters(self, data):
        return (
            not data.empty
            and round(data['Close'].iloc[-1].item(), 2) >= self.min_price
            and data['Volume'].iloc[-1].item() > self.min_volume
        )

    def scan(self):
        result = []
        for ticker in self.fetcher.get_nse_symbols():
            data = self.fetcher.download_data(ticker)

            if self._passes_filters(data):
                check_vcp, start_date, end_date = self.analyzer.analyze(data)

                if check_vcp:
                    result.append({
                        'Stock name': ticker.split('.')[0],
                        'Start date': start_date,
                        'End date': end_date,
                        'Price': data['Close'].iloc[-1].item()
                    })

        return pd.DataFrame(result)


class HistoricalVCPScanner:
    """
    Scans the FULL historical price series of a stock for VCP patterns,
    using the same VCPAnalyzer logic but applied on rolling windows.

    For each detected VCP (start_date -> end_date, i.e. low_idx -> high_3_idx),
    it labels the pattern as 'Good VCP' or 'Bad VCP' based on the price action
    AFTER the pattern completes (the breakout window):

      - 'Good VCP': price moves up by >= breakout_threshold (e.g. 5%) within
                     breakout_window trading days after end_date, without
                     first dropping below the low_2 (last pullback low) by
                     more than stop_loss_threshold.
      - 'Bad VCP' : breakout fails (price does not reach the target,
                     or stop_loss_threshold is breached first).

    The core VCP detection logic (vcp / VCPAnalyzer.analyze) is NOT modified.
    """

    def __init__(self,
                 window_size=150,
                 step=1,
                 breakout_window=20,
                 breakout_threshold=0.07,
                 stop_loss_threshold=0.05,
                 min_price=100,
                 min_volume=150000):
        """
        window_size:        number of bars fed to VCPAnalyzer.analyze() at each step
                             (same as the 150d window used in the original snapshot scan)
        step:               how many bars to slide the window forward each iteration
        breakout_window:    number of trading days after end_date to evaluate outcome
        breakout_threshold: fractional gain (e.g. 0.05 = 5%) that defines a 'Good VCP'
        stop_loss_threshold: fractional drop below low_2 that invalidates the pattern
        min_price/min_volume: same filters as VCPScanner, applied at the END of
                              each window (i.e. at end_date)
        """
        self.window_size = window_size
        self.step = step
        self.breakout_window = breakout_window
        self.breakout_threshold = breakout_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.min_price = min_price
        self.min_volume = min_volume

        self.fetcher = NSEStockFetcher()
        self.analyzer = VCPAnalyzer()

    def _passes_filters(self, window):
        return (
            not window.empty
            and round(window['Close'].iloc[-1].item(), 2) >= self.min_price
            and window['Volume'].iloc[-1].item() > self.min_volume
        )

    def _label_pattern(self, full_data, end_date, low_2_value):
        """
        Determines 'Good VCP' / 'Bad VCP' / 'Unresolved' based on price action
        after end_date.
        """
        # Locate position of end_date in full_data
        end_loc = full_data.index.get_loc(
            full_data[full_data.index.map(lambda x: x.date()) == end_date].index[0]
        )

        future = full_data.iloc[end_loc + 1: end_loc + 1 + self.breakout_window]

        if future.empty:
            return 'Unresolved'

        entry_price = full_data['Close'].iloc[end_loc].item()
        target_price = entry_price * (1 + self.breakout_threshold)
        stop_price = low_2_value * (1 - self.stop_loss_threshold)

        for _, row in future.iterrows():
            low = row['Low'].item() if hasattr(row['Low'], 'item') else row['Low']
            high = row['High'].item() if hasattr(row['High'], 'item') else row['High']

            if low <= stop_price:
                return 'Bad VCP'
            if high >= target_price:
                return 'Good VCP'

        return 'Bad VCP'

    def scan_ticker(self, ticker, full_data=None):
        """
        Scans the full historical data of a single ticker for VCP patterns,
        sliding a window of size `window_size` across the data.

        Returns a list of dicts, one per detected VCP occurrence.
        """
        results = []

        if full_data is None:
            full_data = self.fetcher.download_data(ticker, period='5y', interval='1d')

        if full_data.empty or len(full_data) < self.window_size:
            return results

        seen_end_dates = set()

        for start in range(0, len(full_data) - self.window_size + 1, self.step):
            window = full_data.iloc[start: start + self.window_size].copy()

            if not self._passes_filters(window):
                continue

            check_vcp, start_date, end_date = self.analyzer.analyze(window)

            if not check_vcp:
                continue

            # Avoid duplicate detections of the same pattern as the window slides
            if end_date in seen_end_dates:
                continue
            seen_end_dates.add(end_date)

            # low_2 is needed for stop-loss calc; recompute it from the window
            # using the same approach as in analyze() up to data_4
            low = window['Low'].min().item()
            low_idx = window[np.isclose(window['Low'], low)].index[0]
            data_1 = window.loc[low_idx:]
            high_1 = data_1['High'].max().item()
            high_1_idx = data_1[np.isclose(data_1['High'], high_1)].index[0]
            data_2 = data_1.loc[high_1_idx:]
            low_1 = data_2['Low'].min().item()
            low_1_idx = data_2[np.isclose(data_2['Low'], low_1)].index[0]
            data_3 = data_2.loc[low_1_idx:]
            high_2 = data_3['High'].max().item()
            high_2_idx = data_3[np.isclose(data_3['High'], high_2)].index[0]
            data_4 = window.loc[high_2_idx:]
            low_2 = data_4['Low'].min().item()

            label = self._label_pattern(full_data, end_date, low_2)

            results.append({
                'Stock name': ticker.split('.')[0],
                'Start date': start_date,
                'End date': end_date,
                'Price at end': window['Close'].iloc[-1].item(),
                'Label': label
            })

        return results

    def scan(self, tickers=None):
        """
        Scans multiple tickers for historical VCP patterns.

        tickers: optional list of NSE tickers (e.g. ['RELIANCE.NS', 'TCS.NS']).
                 If None, fetches the full NSE symbol list.

        Returns a DataFrame with columns:
            Stock name, Start date, End date, Price at end, Label
        """
        if tickers is None:
            tickers = self.fetcher.get_nse_symbols()

        all_results = []

        for ticker in tickers:
            try:
                ticker_results = self.scan_ticker(ticker)
                all_results.extend(ticker_results)
            except Exception as e:
                print(f"Skipping {ticker} due to error: {e}")
                continue

        df = pd.DataFrame(all_results)
        if not df.empty:
            df = df.sort_values('End date').groupby(['Stock name', 'Start date'], as_index=False).last()
            df.to_csv('vcp_results.csv', index=False)
        return df


if __name__ == "__main__":
    # Example: scan a small set of tickers for historical VCP patterns
    hist_scanner = HistoricalVCPScanner(
        window_size=150,
        step=1,
        breakout_window=20,
        breakout_threshold=0.07,
        stop_loss_threshold=0.05
    )

    df = hist_scanner.scan()
    print(df)