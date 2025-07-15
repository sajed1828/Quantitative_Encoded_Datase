import os
import asyncio
import time
import aiohttp
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TIINGO_API_KEY = os.getenv("TIINGO_API_KEY") or "06d158eee142ca6dcc1f3169577e269043070750"
now = datetime.now().strftime('%Y-%m-%d')
print(now)

class data_source:
    def __init__(self, symbols, start_date="2023-01-01", end_date=now, freq="1min", delay=1.5, save_csv=False):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.delay = delay
        self.save_csv = save_csv
        self.base_url = "https://api.tiingo.com/iex/{symbol}/prices"

    def _get_url(self, symbol):
        return (
            f"{self.base_url.format(symbol=symbol)}"
            f"?startDate={self.start_date}&endDate={self.end_date}"
            f"&resampleFreq={self.freq}&columns=open,high,low,close,volume"
        )

    async def _fetch(self, session, symbol):
        url = self._get_url(symbol)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {TIINGO_API_KEY}"
        }

        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    df = pd.DataFrame(data)
                    if not df.empty:
                        df["date"] = pd.to_datetime(df["date"])
                        df.set_index("date", inplace=True)
                        df["symbol"] = symbol
                        if self.save_csv:
                            os.makedirs("data", exist_ok=True)
                            df.to_csv(f"data/{symbol}_{self.start_date}_to_{self.end_date}.csv")
                        return df
                    else:
                        print(f" No data for {symbol}")
                else:
                    print(f"Failed to fetch {symbol}: {response.status}")
        except Exception as e:
            print(f" Exception for {symbol}: {e}")
        return None

    async def fetch_all(self):
        async with aiohttp.ClientSession() as session:
            results = []
            for symbol in self.symbols:
                df = await self._fetch(session, symbol)
                if df is not None:
                    results.append(df)
                await asyncio.sleep(self.delay)
            return pd.concat(results) if results else pd.DataFrame()

    def run(self):
        time.sleep(2.5)
        return asyncio.get_event_loop().run_until_complete(self.fetch_all())


