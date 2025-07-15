import pandas as pd
import numpy as np
import ta

class alpha_factor:
    def __init__(self):
        pass

    @staticmethod
    def validate_data(df):
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df

    @staticmethod
    def preprocess_data(df):
        df = alpha_factor.validate_data(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df

    @staticmethod
    def add_momentum_indicators(df):
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['stoch_rsi'] = ta.momentum.StochRSIIndicator(df['close']).stochrsi()
        df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
        df['awesome_osc'] = ta.momentum.AwesomeOscillatorIndicator(df['high'], df['low']).awesome_oscillator()
        df['wr'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        return df

    @staticmethod
    def add_trend_indicators(df):
        macd = ta.trend.MACD(df['close'])
        df['macd_line'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        return df

    @staticmethod
    def add_volatility_indicators(df):
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_bbm'] = bb.bollinger_mavg()
        df['bb_bbh'] = bb.bollinger_hband()
        df['bb_bbl'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_percent'] = bb.bollinger_pband()

        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        return df

    @staticmethod
    def add_volume_indicators(df):
        df['on_balance_volume'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['acc_dist_index'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        return df

    @staticmethod
    def add_returns(df):
        df['returns'] = df['close'].pct_change()
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return df

    @staticmethod
    def ta_factor_indcators(df):
        df = pd.DataFrame(df, columns=['close', 'high', 'low', 'open', 'volume'])
        df = alpha_factor.preprocess_data(df)

        df = alpha_factor.add_momentum_indicators(df)
        df = alpha_factor.add_trend_indicators(df)
        df = alpha_factor.add_volatility_indicators(df)
        df = alpha_factor.add_volume_indicators(df)
        df = alpha_factor.add_returns(df)

        return df.dropna()


