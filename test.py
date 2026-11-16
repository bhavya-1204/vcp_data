import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import requests
from io import StringIO

# Suppress warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

def calculate_ema(df, period):
    """
    Calculate Exponential Moving Average (EMA) for a given period.
    Returns None if insufficient data.
    """
    if len(df) < period:
        return None
    return df['Close'].ewm(span=period, adjust=False).mean()

def vcp(data):
  ema_result = calculate_ema(data, 9)
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

  # print(data['09_ema'].iloc[index_number].item())
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
  # Fixed: Use data_4 for the boolean mask to avoid length mismatch error
  high_3_idx = data_5[np.isclose(data_5['High'], high_3)].index[0]
  index_number_5 = data.index.get_loc(high_3_idx)

  high_3_close = data_5.loc[high_3_idx, 'Close'].item()
  high_3_09 = data['09_ema'].loc[high_3_idx].item()

  if high_3_close < high_3_09:
    return False, None, None

  if high_3 > high_2:
    return False, None, None

  if index_number == index_number_1 or index_number_1 == index_number_2 or index_number_2 == index_number_3 or index_number_3 == index_number_4 or index_number_4 == index_number_5:
    return False, None, None

  distance_1 = (index_number_1 - index_number)
  distance_2 = (index_number_2 - index_number_1)
  distance_3 = (index_number_3 - index_number_2)
  distance_4 = (index_number_4 - index_number_3)
  distance_5 = (index_number_5 - index_number_4)

  # print(low_idx.date())
  # print(high_1_idx.date())
  # print(low_1_idx.date())
  # print(high_2_idx.date())
  # print(low_2_idx.date())
  # print(high_3_idx.date())

  # print(distance_1)
  # print(distance_2)
  # print(distance_3)
  # print(distance_4)
  # print(distance_5)

  distances = [distance_1, distance_2, distance_3, distance_4, distance_5]

  if not all(3 <= d <= 20 for d in distances):
    return False, None, None

  return True, low_idx.date(), high_3_idx.date()

def index():
  result = []
  session = requests.Session()

  headers = {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "Accept-Language": "en-US,en;q=0.9",
      "Referer": "https://www.nseindia.com/"
  }

  session.get("https://www.nseindia.com", headers=headers)  # First hit homepage
  response = session.get(
      "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
      headers=headers
  )
  symbol_name_csv = pd.read_csv(StringIO(response.text))
  nse_stock = [symbol + '.NS' for symbol in symbol_name_csv['SYMBOL']]

  for ticker in nse_stock:
    # data = yf.download(ticker, period='150d', interval='1d', progress=False).round(2)
    for i in range(1, 16):

      data = yf.download(ticker, start=f'2025-11-{i}', end=f'2026-11-{i}', interval='1d', progress=False).round(2)
      # print(data)
      
      if not data.empty and round(data['Close'].iloc[-1].item(),2) >= 100 and data['Volume'].iloc[-1].item() > 1000:
        check_vcp, start_date, end_date = vcp(data)

        if check_vcp:
          result.append({
              'Stock name' : ticker.split('.')[0],
              'Start date' : start_date,
              'End date' : end_date,
              'Price' : data['Close'].iloc[-1].item()
          })
  return pd.DataFrame(result)

if __name__ == "__main__":
  stock = index()
  # stock_df = pd.DataFrame(stock, columns=['Stock Name'])
  # print(stock_df)