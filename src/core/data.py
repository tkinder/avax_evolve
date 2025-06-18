import pandas as pd
import os
import requests
from datetime import datetime, timedelta

CACHE_PATH = "./cache/avax_data.csv"
API_KEY = '4c322523f98c6c20dbe789194197dafac7329ec5a7dc378503118d443e867c2b'

def fetch_historical_data(refresh=False) -> pd.DataFrame:
    if os.path.exists(CACHE_PATH) and not refresh:
        return pd.read_csv(CACHE_PATH, index_col='time', parse_dates=True)

    end_date = pd.Timestamp.today() - pd.Timedelta(days=1)
    url = (
        f"https://min-api.cryptocompare.com/data/v2/histoday?"
        f"fsym=AVAX&tsym=USD&limit=1080&toTs={int(end_date.timestamp())}&api_key={API_KEY}"
    )

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()['Data']['Data']
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    df = df.rename(columns={
        'close': 'close',
        'high': 'high',
        'low': 'low',
        'open': 'open',
        'volumefrom': 'volumefrom',
        'volumeto': 'volumeto'
    })

    numeric_columns = ['close', 'high', 'low', 'open', 'volumefrom', 'volumeto']
    df[numeric_columns] = df[numeric_columns].astype(float)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_csv(CACHE_PATH)

    return df
