import requests
import pandas as pd
import os
import time
import pandas as pd
from tiingo import TiingoClient

TIINGO_API_KEY = "7cf276926816ac91115ead16e29b690f8a75b32b"  # <-- Replace with real key
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "IBM", "ORCL",
    "CRM", "ADBE", "INTU", "NOW", "WDAY", "TEAM", "DOCU", "MDB", "SNOW", "SHOP",
    "PANW", "FTNT", "CRWD", "OKTA", "ZS",
    "AMD", "INTC", "AVGO", "TXN", "MU", "ASML", "TSM", "NXPI", "QCOM",
    "NFLX", "SNAP", "UBER", "LYFT", "SPOT", "RBLX",
    "ABNB", "AMAT", "APP", "ARM", "ADI", "CDNS", "CDW", "CSCO", "CTSH", "KLAC",
    "LRCX", "MCHP", "MELI", "MSTR", "NET", "PDD", "PLTR", "SNPS", "EA", "EBAY",
    "EXPE", "BIDU", "NTES", "DDOG", "COST", "XEL", "MNST", "ON", "PYPL", "ATVI",
    "ADSK", "ANSS", "GFS", "TTD", "ADP", "ISRG", "IDXX", "ZSAN", "ZSIT", "ZSDS",
    "ZM", "DOCN", "HUBS", "BILL", "OKTA", "PATH", "ESTC", "FSLY", "RNG", "TWLO",
    "CRNC", "VRNS", "SPLK", "VERX"
]

START_DATE = "2018-01-01"
FREQUENCY = "daily"
OUTPUT_DIR = "tiingo_data"
REQUEST_DELAY = 1.5

config = {
    'api_key': TIINGO_API_KEY
}
client = TiingoClient(config)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for ticker in TICKERS:
    try:
        print(f"Downloading {ticker}...")
        df = client.get_dataframe(
            ticker,
            startDate=START_DATE,
            frequency=FREQUENCY
        )
        output_path = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
        df.to_csv(output_path)
        print(f"Saved to {output_path}")
        time.sleep(REQUEST_DELAY)
    except Exception as e:
        print(f"Error with {ticker}: {e}")