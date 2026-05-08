"""
Strateji v2 — EMA 21/55 + 200 Rejim + RSI + ADX
==================================================

Giriş Koşulları (LONG)
-----------------------
  1. Fiyat 200h EMA ÜSTÜNDE (boğa rejimi)
  2. EMA 21, EMA 55'i YUKARI keser (golden cross)
  3. RSI(14) ∈ [38, 72]  — aşırı alım/satım filtresi
  4. ADX(14) ≥ 20        — yeterli trend gücü

Giriş Koşulları (SHORT)
-----------------------
  1. Fiyat 200h EMA ALTINDA (ayı rejimi)
  2. EMA 21, EMA 55'i AŞAĞI keser (death cross)
  3. 100 - RSI(14) ∈ [28, 62]
  4. ADX(14) ≥ 20

Çıkış
-----
  - Take-Profit : giriş + 3.5 × ATR
  - Stop-Loss   : giriş − 1.5 × ATR  (2.3:1 RR)
  - Erken çıkış: MA tersine kesişirse pozisyon kapatılır
"""

import pandas as pd
from config import (
    MA_FAST, MA_SLOW, MA_TREND, MA_TYPE,
    ATR_PERIOD, ADX_PERIOD, ADX_THRESHOLD,
    RSI_PERIOD, RSI_LONG_MIN, RSI_LONG_MAX,
    RSI_SHORT_MIN, RSI_SHORT_MAX,
)


# ---------------------------------------------------------------------------
# Gösterge hesaplama
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    dmp = (h - h.shift()).clip(lower=0)
    dmn = (l.shift() - l).clip(lower=0)
    both = dmp > dmn
    dmp  = dmp.where(both, 0)
    dmn  = dmn.where(~both, 0)

    atr_  = _atr(df, period)
    dip   = 100 * dmp.ewm(span=period, adjust=False).mean() / atr_.replace(0, 1e-9)
    din   = 100 * dmn.ewm(span=period, adjust=False).mean() / atr_.replace(0, 1e-9)
    dx    = 100 * (dip - din).abs() / (dip + din).replace(0, 1e-9)
    return dx.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Gösterge ekleme
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    ma_fn = _ema if MA_TYPE == "ema" else _sma

    df = df.copy()
    df["ma_fast"]  = ma_fn(df["close"], MA_FAST)
    df["ma_slow"]  = ma_fn(df["close"], MA_SLOW)
    df["ma_trend"] = ma_fn(df["close"], MA_TREND)
    df["atr"]      = _atr(df, ATR_PERIOD)
    df["rsi"]      = _rsi(df["close"], RSI_PERIOD)
    df["adx"]      = _adx(df, ADX_PERIOD)

    # Crossover: +1 = fast yukarı kesti, -1 = aşağı kesti
    df["cross"] = 0
    prev_above  = df["ma_fast"].shift(1) > df["ma_slow"].shift(1)
    curr_above  = df["ma_fast"] > df["ma_slow"]

    # Rejim filtreleri
    bull     = df["close"] > df["ma_trend"]
    bear     = df["close"] < df["ma_trend"]
    trending = df["adx"] >= ADX_THRESHOLD

    rsi_long_ok  = (df["rsi"] >= RSI_LONG_MIN)  & (df["rsi"] <= RSI_LONG_MAX)
    rsi_short_ok = ((100 - df["rsi"]) >= RSI_SHORT_MIN) & \
                   ((100 - df["rsi"]) <= RSI_SHORT_MAX)

    long_cond  = ~prev_above & curr_above & bull & trending & rsi_long_ok
    short_cond = prev_above & ~curr_above & bear & trending & rsi_short_ok

    df.loc[long_cond,  "cross"] =  1   # BUY
    df.loc[short_cond, "cross"] = -1   # SELL

    return df


# ---------------------------------------------------------------------------
# Sinyal üretimi
# ---------------------------------------------------------------------------

def get_signal(df: pd.DataFrame) -> dict:
    """
    Son kapanan mumun sinyalini döndürür.

    Returns
    -------
    dict:
        signal    : 'buy' | 'sell' | 'none'
        price     : giriş fiyatı
        atr       : ATR değeri
        ma_fast   : hızlı EMA
        ma_slow   : yavaş EMA
        ma_trend  : rejim EMA
        rsi       : RSI değeri
        adx       : ADX değeri
        bull      : boğa rejiminde mi?
    """
    if df is None or len(df) < MA_TREND + 2:
        return {"signal": "none"}

    df = add_indicators(df)
    row = df.iloc[-2]   # son KAPANAN mum

    result = {
        "signal"  : "none",
        "price"   : float(row["close"]),
        "atr"     : float(row["atr"]),
        "ma_fast" : float(row["ma_fast"]),
        "ma_slow" : float(row["ma_slow"]),
        "ma_trend": float(row["ma_trend"]),
        "rsi"     : float(row["rsi"]),
        "adx"     : float(row["adx"]),
        "bull"    : bool(row["close"] > row["ma_trend"]),
    }

    if row["cross"] == 1:
        result["signal"] = "buy"
    elif row["cross"] == -1:
        result["signal"] = "sell"

    return result


def should_exit_early(df: pd.DataFrame, side: str) -> bool:
    """
    Açık pozisyon için MA tersine kesişirse erken çıkış sinyali verir.
    """
    if df is None or len(df) < MA_SLOW + 2:
        return False

    df = add_indicators(df)
    row = df.iloc[-2]

    if side == "buy"  and row["cross"] == -1:
        return True
    if side == "sell" and row["cross"] == 1:
        return True
    return False
