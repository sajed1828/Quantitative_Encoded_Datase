import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as web
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
#import quandl
import yfinance as yf 
import warnings
warnings.filterwarnings("ignore")

# Set the start and end dates for the data

tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']

prices = yf.download(
    tickers,
    start="2000-01-01",
    end="2018-12-31",
    group_by='ticker',
    auto_adjust=True,   
    threads=True
   )

class data_factors:
  def __init__(self):
        pass
 
  def monthely(self, prices, tickers):
    monthly =  pd.DataFrame()
    for ticker in tickers:
      monthly = (
      prices[ticker]['Close']
      .resample('ME')    
      .last()
      )
    
    monthe_price = monthly.copy()
    lags = [1,2,3,6,9,12]
    outlier = 0.01
    data = pd.DataFrame()
    for lag in lags:
        rtn = (monthe_price
        .pct_change(lag)
        .pipe(lambda x: x.clip(lower=x.quantile(outlier), upper=x.quantile(1-outlier)))
        .add(1)
        .pow(1/lag)
        .sub(1)
        )
        data[f'return_{lag}m'] = rtn
        
    return data

class Stratige:
  def __init__(self):
     pass
  
  def PandasRollingOLS(self):
    
    
    return
  
  def RollingOLS(self):
    
    
    return 
  
  def OLS(self):
    
    
    return 

