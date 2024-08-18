import quandl
import pandas as pd
from datetime import datetime, timedelta

def fetch_quandl_data(symbol, api_key):
    quandl.ApiConfig.api_key = api_key
    try:
        data = quandl.get(f"EOD/{symbol}", start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
        data = data['Adj_Close'].resample('MS').first()
        return data.rename(symbol)
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")
        return pd.Series(name=symbol)

api_key_quandl = 'kfRwmRA9dUzf64Gxx5yQ'
end_date = datetime.now()
start_date = end_date - timedelta(days=14*30)

symbols = ['QQQ', 'SPY', 'DIA']
data = {symbol: fetch_quandl_data(symbol, api_key_quandl) for symbol in symbols}
combined_df = pd.DataFrame(data)

print(combined_df)
