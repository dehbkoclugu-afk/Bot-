"""
MA Crossover Strategy  (v2 — ADX trend filter)
-----------------------------------------------
Signal  : Fast MA (5h) crosses Slow MA (12h)
Buy     : fast crosses ABOVE slow  (golden cross)
Sell    : fast crosses BELOW slow  (death cross)
Filter  : ADX > 20  → only trade when trend is strong enough
          (avoids false signals in sideways/choppy markets)

ATR is used for dynamic Take-Profit and Stop-Loss levels.
"""

import pandas as pd
from config import MA_FAST, MA_SLOW, MA_TYPE, ATR_PERIOD

ADX_PERIOD    = 14
ADX_THRESHOLD = 20   # ignore signals when ADX < this value


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's ADX — measures trend strength (0-100, >20 = trending)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)

    dm_plus  = (high - prev_high).clip(lower=0)
    dm_minus = (prev_low - low).clip(lower=0)
    # When both positive, keep the larger; zero the smaller
    both = dm_plus > dm_minus
    dm_plus  = dm_plus.where(both, 0)
    dm_minus = dm_minus.where(~both, 0)

    atr_raw  = _atr(df, period)
    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean()  / atr_raw.replace(0, 1e-9)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr_raw.replace(0, 1e-9)

    dx_denom = (di_plus + di_minus).replace(0, 1e-9)
    dx  = 100 * (di_plus - di_minus).abs() / dx_denom
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA_FAST, MA_SLOW, ATR and ADX columns to OHLCV dataframe."""
    ma_fn = _ema if MA_TYPE == "ema" else _sma

    df = df.copy()
    df["ma_fast"] = ma_fn(df["close"], MA_FAST)
    df["ma_slow"] = ma_fn(df["close"], MA_SLOW)
    df["atr"]     = _atr(df, ATR_PERIOD)
    df["adx"]     = _adx(df, ADX_PERIOD)

    # Crossover flags: +1 = fast crossed above slow, -1 = crossed below
    df["cross"] = 0
    prev_above = df["ma_fast"].shift(1) > df["ma_slow"].shift(1)
    curr_above = df["ma_fast"] > df["ma_slow"]

    # Apply ADX filter: signal only when trend is strong
    trending = df["adx"] >= ADX_THRESHOLD

    df.loc[~prev_above & curr_above & trending, "cross"] = 1   # golden cross → BUY
    df.loc[prev_above & ~curr_above & trending, "cross"] = -1  # death  cross → SELL

    return df


def get_signal(df: pd.DataFrame) -> dict:
    """
    Return the signal for the latest *closed* candle.

    Returns
    -------
    dict with keys:
        signal  : 'buy' | 'sell' | 'none'
        price   : close price of the signal candle
        atr     : ATR value at signal candle
        ma_fast : fast MA value
        ma_slow : slow MA value
    """
    if df is None or len(df) < MA_SLOW + 2:
        return {"signal": "none"}

    df = add_indicators(df)

    # Use the second-to-last row (last *closed* candle; last row may be live)
    row = df.iloc[-2]

    result = {
        "signal" : "none",
        "price"  : float(row["close"]),
        "atr"    : float(row["atr"]),
        "ma_fast": float(row["ma_fast"]),
        "ma_slow": float(row["ma_slow"]),
    }

    if row["cross"] == 1:
        result["signal"] = "buy"
    elif row["cross"] == -1:
        result["signal"] = "sell"

    return result
