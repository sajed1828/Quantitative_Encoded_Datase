import numpy as np
import pandas as pd


# Define helper functions for alpha formulas
def rank(series):
    return series.rank(pct=True)

def ts_rank(series, window):
    return series.rolling(int(window)).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

def delta(series, period=1):
    return series.diff(period)

def delay(series, period=1):
    return series.shift(period)

def correlation(x, y, window):
    return x.rolling(int(window)).corr(y)

def covariance(x, y, window):
    return x.rolling(int(window)).cov(y)

def signed_power(series, exponent):
    return np.sign(series) * (np.abs(series) ** exponent)

def stddev(series, window):
    return pd.Series(series).rolling(int(window)).std()

def sum_(series, window):
    return series.rolling(int(window)).sum()

def ts_min(series, window):
    return pd.Series(series).rolling(int(window)).min()

def ts_max(series, window):
    return pd.Series(series).rolling(int(window)).max()

def decay_linear(series, window):
    weights = np.arange(1, int(window) + 1)
    return series.rolling(int(window)).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def scale(series):
    return series / np.sum(np.abs(series))

def product(series):
    return pd.Series(series).prod()

def sign(series):
    return np.sign(series)

def log(series):
    return np.log(series)

def sum_series(series, window):
    return series.rolling(int(window)).sum()

def Ts_Rank(series, window):
    return series.rolling(int(window)).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

def IndNeutralize(series, group):
    return series.groupby(group).transform(lambda x: x - x.mean())

def min_(series, window):
    return series.rolling(int(window)).min()

def ts_argmax(series, window):
    return series.rolling(int(window)).apply(np.argmax) / window


# Example alpha formula implementation
# Alpha Factors
class Alpha_Zero:
 def __init__(self):
     super().__init__()

 @staticmethod       
 def alpha_1(df):
    condition = df['returns'] < 0
    expr = np.where(condition, stddev(df['returns'], 20), df['close'])
    ranked = rank(ts_max(signed_power(expr, 2), 5))
    return ranked - 0.5

 @staticmethod
 def alpha_2(df):
    log_volume = np.log(df['volume'].replace(0, np.nan))  # Avoid log(0)
    delta_log_vol = delta(log_volume, 2)
    ranked_delta_log_vol = rank(delta_log_vol)

    price_change = (df['close'] - df['open']) / df['open']
    ranked_price_change = rank(price_change)

    return -1 * correlation(ranked_delta_log_vol, ranked_price_change, 6)

 @staticmethod
 def alpha_3(df):
    return (-1 * correlation(rank(df['close']), rank(df['volume']), 10))

 @staticmethod
 def alpha_4(df):
    return (-1 * ts_rank(rank(df['low']), 9))

 @staticmethod
 def alpha_5(df):
    vwap = df['vwap'].rolling(window=10).mean()
    return (rank((df['open'] - vwap)) * (-1 * rank(df['close'] - df['vwap']).abs()))

 @staticmethod
 def alpha_6(df):
    return (-1 * correlation(df['open'] , df['close'], 10))

 @staticmethod
 def alpha_7(df):
    adv20 = df['volume'].rolling(window=20).mean()
    delat_close_7 = delta(df['close'], 7)
    ts_r = ts_rank(abs(delat_close_7), 60)
    
    return pd.Series( np.where(
            adv20 < df['volume'],
            (-1 * ts_r * np.sign(delat_close_7)),
            -1.0
        ),
        index=df.index
    )

 @staticmethod
 def alpha_8(df):
    term = df['open'].rolling(5).sum() * df['returns'].rolling(5).sum()
    return -1 * rank(term - delay(term, 10))

 @staticmethod
 def alpha_9(df):
    delta_close = delta(df['close'], 1)
    return np.where(
        ts_min(delta_close, 5) > 0,
        delta_close,
        np.where(ts_max(delta_close, 5) < 0, delta_close, -1 * delta_close)
    )

 @staticmethod
 def alpha_10(df):
    delta_close = delta(df['close'], 1)
    cond = np.where(
        ts_min(delta_close, 4) > 0,
        delta_close,
        np.where(ts_max(delta_close, 4) < 0, delta_close, -1 * delta_close)
    )
    return rank(pd.Series(cond, index=df.index))

 @staticmethod
 def alpha_11(df):
    diff = df['vwap'] - df['close']
    return (rank(ts_max(diff, 3)) + rank(ts_min(diff, 3))) * rank(delta(df['volume'], 3))

 @staticmethod
 def alpha_12(df):
    return sign(delta(df['volume'], 1)) * (-1 * delta(df['close'], 1))

 @staticmethod
 def alpha_13(df):
    return -1 * rank(covariance(rank(df['close']), rank(df['volume']), 5))

 @staticmethod
 def alpha_14(df):
    return (-1 * rank(delta(df['returns'], 3))) * correlation(df['open'], df['volume'], 10)

 @staticmethod
 def alpha_15(df):
    corrs = rank(correlation(rank(df['high']), rank(df['volume']), 3))
    return -1 * corrs.rolling(window=3).sum()

 @staticmethod
 def alpha_16(df):
    return -1 * rank(covariance(rank(df['high']), rank(df['volume']), 5))

 @staticmethod
 def alpha_17(df):
    return -1 * rank(covariance(rank(df['close']), rank(df['volume']), 5))

 @staticmethod
 def alpha_18(df):
    close_open = df['close'] - df['open']
    term = stddev(abs(close_open), 5) + close_open
    return -1 * rank(term + correlation(df['close'], df['open'], 10))

 @staticmethod
 def alpha_19(df):
   term = (df['close'] - delay(df['close'], 7)) + delta(df['close'], 7)
   return -1 * sign(term) * (1 + rank(1 + sum_(df['returns'], 250)))

 @staticmethod
 def alpha_20(df):
    term = (df['close'] - delay(df['close'], 7)) + delta(df['close'], 7)
    return -1 * sign(term) * (1 + rank(1 + df['returns'].rolling(250).sum()))

 @staticmethod
 def alpha_21(df):
    avg_8 = df['close'].rolling(8).mean()
    std_8 = stddev(df['close'], 8)
    avg_2 = df['close'].rolling(2).mean()
    vol_ratio = df['volume'] / df['volume'].rolling(20).mean()
    return np.where(
        (avg_8 + std_8) < avg_2, -1,
        np.where(avg_2 < (avg_8 - std_8), 1,
                 np.where((vol_ratio > 1) | (vol_ratio == 1), 1, -1))
    )

 @staticmethod
 def alpha_22(df):
    return -1 * (delta(correlation(df['high'], df['volume'], 5), 5) * rank(stddev(df['close'], 20)))

 @staticmethod
 def alpha_23(df):
    return np.where(
        (df['high'].rolling(20).mean() < df['high']),
        -1 * delta(df['high'], 2),
        0)

 @staticmethod
 def alpha_24(df):
   mean_close_100 = df['close'].rolling(100).mean()
   delta_mean = delta(mean_close_100, 100)
   delay_close = delay(df['close'], 100)
   ratio = delta_mean / delay_close
   cond = (ratio < 0.05) | (ratio == 0.05)
   return np.where(
      cond,
      -1 * (df['close'] - ts_min(df['close'], 100)),
      -1 * delta(df['close'], 3))

 @staticmethod 
 def alpha_25(df):
    adv20 = df['volume'].rolling(window=20).mean()
    return rank(((-1 * df['returns']) * adv20 * df['vwap'] * (df['high'] - df['close'])))

 @staticmethod
 def alpha_26(df):
    corr_series = correlation(rank(df['volume']), rank(df['vwap']), 6)
    sum_corr = sum_(corr_series, 2) / 2.0
    rank_sum_corr = rank(sum_corr)
    return np.where(rank_sum_corr > 0.5, -1, 1)

 @staticmethod
 def alpha_27(df):
    adv20 = df['volume'].rolling(window=20).mean()
    corr_val = correlation(adv20, df['low'], 5)
    middle = (df['high'] + df['low']) / 2
    return scale(corr_val + middle - df['close'])
 
 @staticmethod
 def alpha_28(df):
   # Alpha#28: scale((correlation(adv20, low, 5) + ((high + low) / 2 - close)))
   adv20 = df['volume'].rolling(window=20).mean()
   corr_val = correlation(adv20, df['low'], 5)
   middle = (df['high'] + df['low']) / 2
   return scale(corr_val + middle - df['close'])

 @staticmethod
 def alpha_29(df):
    return -1 * ts_max(correlation(ts_rank(df['volume'], 5), ts_rank(df['high'], 5), 5), 3)

 @staticmethod
 def alpha_30(df):
    cond = (sign(df['close'] - delay(df['close'], 1)) +
            sign(delay(df['close'], 1) - delay(df['close'], 2)) +
            sign(delay(df['close'], 2) - delay(df['close'], 3)))
    return ((1.0 - rank(cond)) * sum_(df['volume'], 5)) / sum_(df['volume'], 20)

 @staticmethod
 def alpha_31(df):
    adv20 = df['close'].rolling(window=20).mean()
    part1 = rank(rank(rank(decay_linear(-1 * rank(rank(delta(df['close'], 10))), 10))))
    part2 = rank(-1 * delta(df['close'], 3))
    part3 = sign(scale(correlation(adv20, df['low'], 12)))
    return part1 + part2 + part3

 @staticmethod
 def alpha_32(df):
    part1 = scale((sum_(df['close'], 7) / 7) - df['close'])
    part2 = 20 * scale(correlation(df['vwap'], delay(df['close'], 5), 230))
    return part1 + part2
 
 @staticmethod
 def alpha_33(df):
        # Alpha#33: rank((-1 * ((1 - (open / close))^1)))
        factor = -1 * ((1 - (df['open'] / df['close'])) ** 1)
        return rank(factor)

 @staticmethod
 def alpha_34(df):
        # Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
        std_2 = stddev(df['returns'], 2)
        std_5 = stddev(df['returns'], 5)
        delta_1 = delta(df['close'], 1)

        rank_std_ratio = rank(std_2 / std_5)
        rank_delta_close = rank(delta_1)

        factor = (1 - rank_std_ratio) + (1 - rank_delta_close)
        return rank(factor)

 @staticmethod
 def alpha_35(df):
        # Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
        ts_rank_volume = ts_rank(df['volume'], 32)
        ts_rank_price_range = ts_rank((df['close'] + df['high'] - df['low']), 16)
        ts_rank_returns = ts_rank(df['returns'], 32)

        factor = ts_rank_volume * (1 - ts_rank_price_range) * (1 - ts_rank_returns)
        return factor

 @staticmethod
 def alpha_36(df):
    adv20 = df['volume'].rolling(window=20).mean()
    part1 = 2.21 * rank(correlation(df['close'] - df['open'], delay(df['volume'], 1), 15))
    part2 = 0.7 * rank(df['open'] - df['close'])
    part3 = 0.73 * rank(ts_rank(delay(-1 * df['returns'], 6), 5))
    part4 = rank(abs(correlation(df['vwap'], adv20, 6)))
    part5 = 0.6 * rank((sum_(df['close'], 200) / 200 - df['open']) * (df['close'] - df['open']))
    return part1 + part2 + part3 + part4 + part5

 @staticmethod
 def alpha_37(df):
        # Alpha#37: rank(correlation(delay((open - close), 1), close, 200)) + rank(open - close)
        delayed_diff = delay(df['open'] - df['close'], 1)
        corr_val = correlation(delayed_diff, df['close'], 200)
        return rank(corr_val) + rank(df['open'] - df['close'])

 @staticmethod
 def alpha_38(df):
        # Alpha#38: (-1 * rank(ts_rank(close, 10))) * rank(close / open)
        ts_rk = ts_rank(df['close'], 10)
        return (-1 * rank(ts_rk)) * rank(df['close'] / df['open'])


 @staticmethod
 def alpha_39(df):
    adv20 = df['volume'].rolling(window=20).mean()
    part1 = rank(correlation(delay(df['open'] - df['close'], 1), df['close'], 200))
    part2 = rank(df['open'] - df['close'])
    part3 = (-1 * rank(ts_rank(df['close'], 10))) * rank(df['close'] / df['open'])
    part4 = (-1 * rank(delta(df['close'], 7) * (1 - rank(decay_linear(df['volume'] / adv20, 9))))) * (1 + rank(sum_(df['returns'], 250)))
    return part1 + part2 + part3 + part4

 @staticmethod
 def alpha_40(df):
    return (-1 * rank(stddev(df['high'], 10))) * correlation(df['high'], df['volume'], 10)

 @staticmethod
 def alpha_41(df):
    return ((df['high'] * df['low']) ** 0.5) - df['vwap']

 @staticmethod
 def alpha_42(df):
    return rank(df['vwap'] - df['close']) / rank(df['vwap'] + df['close'])

 @staticmethod
 def alpha_43(df):
    adv20 = df['volume'].rolling(window=20).mean()
    
    return ts_rank(df['volume'] / adv20, 20) * ts_rank(-1 * delta(df['close'], 7), 8) 

 @staticmethod
 def alpha_44(df):
        return -1 * correlation(df['high'], rank(df['volume']), 5)

 @staticmethod
 def alpha_45(df):
        part1 = rank(sum_(delay(df['close'], 5), 20) / 20)
        part2 = correlation(df['close'], df['volume'], 2)
        part3 = rank(correlation(sum_(df['close'], 5), sum_(df['close'], 20), 2))
        return -1 * part1 * part2 * part3

 @staticmethod
 def alpha_46(df):
        val = ((delay(df['close'], 20) - delay(df['close'], 10)) / 10) - ((delay(df['close'], 10) - df['close']) / 10)
        return np.where(val > 0.25, -1, np.where(val < 0, 1, -1 * (df['close'] - delay(df['close'], 1))))

 @staticmethod
 def alpha_47(df):
        adv20 = df['volume'].rolling(20).mean()
        part1 = (rank(1 / df['close']) * df['volume']) / adv20
        part2 = (df['high'] * rank(df['high'] - df['close'])) / (sum_(df['high'], 5) / 5)
        return (part1 * part2) - rank(df['vwap'] - delay(df['vwap'], 5))

 @staticmethod
 def alpha_48(df):
   """def alpha_48(df):
   num = correlation(delta(df['close'], 1), delta(delay(df['close'], 1), 1), 250) * delta(df['close'], 1) / df['close']
   denom = sum_((delta(df['close'], 1) / delay(df['close'], 1))**2, 250)
   return IndNeutralize(num, df['subindustry']) / denom
   """
   return 
 
 @staticmethod
 def alpha_49(df):
        val = ((delay(df['close'], 20) - delay(df['close'], 10)) / 10) - ((delay(df['close'], 10) - df['close']) / 10)
        return np.where(val < -0.1, 1, -1 * (df['close'] - delay(df['close'], 1)))

 @staticmethod
 def alpha_50(df):
        corr = correlation(rank(df['volume']), rank(df['vwap']), 5)
        return -1 * ts_max(rank(corr), 5)

 @staticmethod
 def alpha_51(df):
        val = ((delay(df['close'], 20) - delay(df['close'], 10)) / 10) - ((delay(df['close'], 10) - df['close']) / 10)
        return np.where(val < -0.05, 1, -1 * (df['close'] - delay(df['close'], 1)))

 @staticmethod
 def alpha_52(df):
        low_min = ts_min(df['low'], 5)
        returns_diff = (sum_(df['returns'], 240) - sum_(df['returns'], 20)) / 220
        return ((-1 * low_min + delay(low_min, 5)) * rank(returns_diff)) * ts_rank(df['volume'], 5)

 @staticmethod
 def alpha_53(df):
        numerator = (df['close'] - df['low']) - (df['high'] - df['close'])
        denominator = df['close'] - df['low']
        return -1 * delta(numerator / denominator, 9)

 @staticmethod
 def alpha_54(df):
        return (-1 * ((df['low'] - df['close']) * (df['open']**5))) / ((df['low'] - df['high']) * (df['close']**5))

 @staticmethod
 def alpha_55(df):
        norm = (df['close'] - ts_min(df['low'], 12)) / (ts_max(df['high'], 12) - ts_min(df['low'], 12))
        return -1 * correlation(rank(norm), rank(df['volume']), 6)

 @staticmethod
 def alpha_56(df):
        df['cap'] = df['vwap'] * df['volume']
        num = sum_(df['returns'], 10)
        denom = sum_(sum_(df['returns'], 2), 3)
        return -1 * (rank(num / denom) * rank(df['returns'] * df['cap']))

 @staticmethod
 def alpha_57(df):
        return -1 * ((df['close'] - df['vwap']) / decay_linear(rank(ts_argmax(df['close'], 30)), 2))

 @staticmethod
 def alpha_56(df): 
  """ def alpha_58(df):
        vwap_neutral = IndNeutralize(df['vwap'], df['sector'])
        corr = correlation(vwap_neutral, df['volume'], 3.92795)
        return -1 * ts_rank(decay_linear(corr, 7.89291), 5.50322)
  """
  return

 @staticmethod
 def alpha_57(df): 
  """def alpha_59(df):
        neutral = IndNeutralize(df['vwap'] * 0.728317 + df['vwap'] * (1 - 0.728317), df['industry'])
        corr = correlation(neutral, df['volume'], 4.25197)
        return -1 * ts_rank(decay_linear(corr, 16.2289), 8.19648)
  """
  return 
 
 @staticmethod
 def alpha_58(df):
    return

 @staticmethod
 def alpha_59(df):
    return

 @staticmethod
 def alpha_60(df):
        val = (((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])) * df['volume']
        return -1 * ((2 * scale(rank(val))) - scale(rank(ts_argmax(df['close'], 10))))
 
 @staticmethod
 def alpha_61(df):
  """ def alpha_61(df):
    adv180 = df['volume'].rolling(window=180).mean()
    return (rank(df['vwap'] - ts_min(df['vwap'], 16.1219)) < rank(correlation(df['vwap'], adv180, 17.9282)))
  """
  return 
 
 @staticmethod
 def alpha_62(df):
  """def alpha_62(df):
    return (
        (rank(correlation(df['vwap'], sum_series(df['adv20'], 22.4101), 9.91009)) <
         rank((rank(df['open']) + rank(df['open'])) < (rank((df['high'] + df['low']) / 2) + rank(df['high'])))) * -1
    )
  """
  return
 
 @staticmethod
 def alpha_63(df):
    """ def alpha_63(close, df, open_, adv180, industry):
    part1 = rank(decay_linear(delta(IndNeutralize(close, industry), 2.25164), 8.22237))
    part2 = rank(decay_linear(
        correlation(df['vwap'] * 0.318108 + open_ * (1 - 0.318108),
                    sum_series(adv180, 37.2467), 13.557), 12.2883))
    return (part1 - part2) * -1
    """
    return 
  
 @staticmethod
 def alpha_64(df):
    adv120 = df['volume'].rolling(window=120).mean()
    return (
        (rank(correlation(sum_series(df['open'] * 0.178404 + df['low'] * (1 - 0.178404), 12.7054),
                          sum_series(adv120, 12.7054), 16.6208)) <
         rank(delta(((df['high'] + df['low']) / 2) * 0.178404 + df['vwap'] * (1 - 0.178404), 3.69741))) * -1
    )

 @staticmethod
 def alpha_65(df):
    adv60 = df['volume'].rolling(window=60).mean()
    return (
        (rank(correlation(df['open'] * 0.00817205 + df['vwap'] * (1 - 0.00817205), sum_series(adv60, 8.6911), 6.40374)) <
         rank(df['open'] - ts_min(df['open'], 13.635))) * -1
    )
 
 @staticmethod
 def alpha_66(df):
    decay1 = rank(decay_linear(delta(df['vwap'], 3.51013), 7.23052))
    decay2 = Ts_Rank(decay_linear(((df['low'] - df['vwap']) / (df['open'] - ((df['high'] + df['low']) / 2))), 11.4157), 6.72611)
    return (decay1 + decay2) * -1

 @staticmethod
 def alpha_67(df): 
  """ def alpha_67(high, vwap, adv20, sector, subindustry):
    r1 = rank(high - ts_min(high, 2.14593))
    r2 = rank(correlation(IndNeutralize(vwap, sector), IndNeutralize(adv20, subindustry), 6.02936))
    return (r1 ** r2) * -1
  """
  return 

 @staticmethod
 def alpha_68(df):
    adv15 =  df['close'].rolling(window= 60).mean()
    return (
        (Ts_Rank(correlation(rank(df['high']), rank(adv15), 8.91644), 13.9333) <
         rank(delta(df['close'] * 0.518371 + df['low'] * (1 - 0.518371), 1.06157))) * -1
    )
 
 @staticmethod
 def alpha_69(df):
  """def alpha_69(df, close, adv20, industry):
    max_delta = rank(ts_max(delta(IndNeutralize(df['vwap'], industry), 2.72412), 4.79344))
    corr = Ts_Rank(correlation(close * 0.490655 + df['vwap'] * (1 - 0.490655), adv20, 4.92416), 9.0615)
    return (max_delta ** corr) * -1
  """
  return

 @staticmethod
 def alpha_70(df): 
  """def alpha_70(vwap, close, adv50, industry):
    return (
        (rank(delta(vwap, 1.29456)) **
         Ts_Rank(correlation(IndNeutralize(close, industry), adv50, 17.8256), 17.9171)) * -1
    )
  """
  return
  
 @staticmethod
 def alpha_71(df):
    """def alpha_71(close, adv180, low, open_, vwap):
    p1 = Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948)
    p2 = Ts_Rank(decay_linear((rank((low + open_ - (vwap + vwap))) ** 2), 16.4662), 4.4388)
    return np.maximum(p1, p2)
    """
    return
 
 @staticmethod
 def alpha_72(df):
    adv40 = df['close'].rolling(window=40).mean()
    numerator = rank(decay_linear(correlation((df['high'] + df['low']) / 2, adv40, 8.93345), 10.1519))
    denominator = rank(decay_linear(correlation(Ts_Rank(df['vwap'], 3.72469), Ts_Rank(df['volume'], 18.5188), 6.86671), 2.95011))
    return numerator / denominator
 
 @staticmethod
 def alpha_73(df):
    part1 = rank(decay_linear(delta(df['vwap'], 4.72775), 2.91864))
    ratio = delta(df['open'] * 0.147155 + df['low'] * (1 - 0.147155), 2.03608) / (df['open'] * 0.147155 + df['low'] * (1 - 0.147155))
    part2 = Ts_Rank(decay_linear(ratio * -1, 3.33829), 16.7411)
    return np.maximum(part1, part2) * -1
 
 @staticmethod
 def colpute_all_alpha_factors(df_sym):
    for i in range(1, 48):
            func = getattr(Alpha_Zero, f"alpha_{i}")
            df_sym[f"alpha_factors_{i}"] = func(df_sym)
    return df_sym    


