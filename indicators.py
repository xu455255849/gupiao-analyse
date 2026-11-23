"""技术指标计算：MA、MACD、RSI、KDJ、BOLL、支撑压力位。"""
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    # 均线
    for n in (5, 10, 20, 60):
        df[f"ma{n}"] = c.rolling(n).mean()

    # MACD (12, 26, 9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd"] = (df["dif"] - df["dea"]) * 2

    # RSI (14)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)

    # KDJ (9, 3, 3)
    low9 = df["low"].rolling(9).min()
    high9 = df["high"].rolling(9).max()
    rsv = (c - low9) / (high9 - low9) * 100
    df["k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    df["d"] = df["k"].ewm(alpha=1 / 3, adjust=False).mean()
    df["j"] = 3 * df["k"] - 2 * df["d"]

    # 布林带 (20, 2)
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    df["boll_mid"] = mid
    df["boll_up"] = mid + 2 * std
    df["boll_low"] = mid - 2 * std

    # 量比：当日成交量 / 5日均量
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma5"].shift(1)

    return df


def support_resistance(df: pd.DataFrame, window: int = 60) -> dict:
    """基于近 window 日的支撑/压力位。"""
    recent = df.tail(window)
    close = df["close"].iloc[-1]
    support = recent["low"].quantile(0.1)
    resistance = recent["high"].quantile(0.9)
    return {
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "pos_in_range": round((close - recent["low"].min()) /
                              max(recent["high"].max() - recent["low"].min(), 1e-9) * 100, 1),
    }
