"""
Strateji v3 — EMA 21/55 Trend + StochRSI Giriş + Trailing Stop
================================================================

Giriş Mantığı (ENTRY_MODE = "stochrsi")
-----------------------------------------
  Adım 1 — TREND: EMA21 > EMA55 (boğa trendi aktif)
  Adım 2 — REJİM: Fiyat > EMA200 (boğa rejimi)
  Adım 3 — ADX ≥ 25 (güçlü trend)
  Adım 4 — StochRSI_K, oversold bölgesine (<0.20) girdi → ardından
            K, D'yi yukarı kesince GİRİŞ (boğa rejiminde)
  Adım 5 — Giriş mumunun hacmi ≥ 20-bar ort. × 1.2

Çıkış
-----
  Trailing ATR Stop: +1.5×ATR'de breakeven, +2.5×ATR'de trail aktif
  Emniyet TP kapağı: 8×ATR
  (Sabit TP ve MA çapraz çıkışı kaldırıldı)
"""

import pandas as pd
from config import (
    MA_FASTEST, MA_FAST, MA_SLOW, MA_TREND, MA_TYPE,
    ATR_PERIOD, ADX_PERIOD, ADX_THRESHOLD,
    RSI_PERIOD,
    SRSI_PERIOD, SRSI_K_SMOOTH, SRSI_D_SMOOTH,
    SRSI_OVERSOLD, SRSI_OVERBOUGHT,
    VOLUME_MA_PERIOD, VOLUME_MIN_MULT,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    SCORE_MIN, SCORE_HIGH, RISK_HIGH, RISK_LOW,
    ST_ATR_PERIOD, ST_MULTIPLIER,
    ENTRY_MODE,
)


# ---------------------------------------------------------------------------
# Göstergeler
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
    return 100 - 100 / (1 + gain / loss.replace(0, 1e-9))


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    dmp = (h - h.shift()).clip(lower=0)
    dmn = (l.shift() - l).clip(lower=0)
    both = dmp > dmn
    dmp  = dmp.where(both, 0)
    dmn  = dmn.where(~both, 0)
    atr_ = _atr(df, period)
    dip  = 100 * dmp.ewm(span=period, adjust=False).mean() / atr_.replace(0, 1e-9)
    din  = 100 * dmn.ewm(span=period, adjust=False).mean() / atr_.replace(0, 1e-9)
    dx   = 100 * (dip - din).abs() / (dip + din).replace(0, 1e-9)
    return dx.ewm(span=period, adjust=False).mean()


def _stoch_rsi(
    series: pd.Series,
    rsi_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI: K ve D serisini döndürür (0-1 aralığında)."""
    rsi     = _rsi(series, rsi_period)
    rsi_min = rsi.rolling(rsi_period).min()
    rsi_max = rsi.rolling(rsi_period).max()
    stoch   = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-9)
    k = stoch.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d


def _macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD çizgisi, sinyal çizgisi ve histogram döndürür."""
    line = _ema(series, fast) - _ema(series, slow)
    sig  = _ema(line, signal)
    hist = line - sig
    return line, sig, hist


def _supertrend(
    df: pd.DataFrame,
    atr_period: int = 14,
    multiplier: float = 7.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend: trend (1=boğa/-1=ayı) ve destek/direnç çizgisi döndürür.
    Destek çizgisi long pozisyon için dinamik SL olarak kullanılır.
    """
    atr   = _atr(df, atr_period)
    hl2   = (df["high"] + df["low"]) / 2
    close = df["close"].values
    upper = (hl2 + multiplier * atr).values
    lower = (hl2 - multiplier * atr).values
    n = len(df)

    fu = upper.copy()
    fl = lower.copy()
    trend   = [1] * n
    support = [0.0] * n

    for i in range(1, n):
        fu[i] = upper[i] if upper[i] < fu[i-1] or close[i-1] > fu[i-1] else fu[i-1]
        fl[i] = lower[i] if lower[i] > fl[i-1] or close[i-1] < fl[i-1] else fl[i-1]

        if trend[i-1] == -1 and close[i] > fu[i-1]:
            trend[i] = 1
        elif trend[i-1] == 1 and close[i] < fl[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

        support[i] = fl[i] if trend[i] == 1 else fu[i]

    return (
        pd.Series(trend,   index=df.index, name="st_trend"),
        pd.Series(support, index=df.index, name="st_support"),
    )


# ---------------------------------------------------------------------------
# Gösterge ekleme
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    ma_fn = _ema if MA_TYPE == "ema" else _sma

    df = df.copy()
    df["ma_fastest"] = ma_fn(df["close"], MA_FASTEST)
    df["ma_fast"]    = ma_fn(df["close"], MA_FAST)
    df["ma_slow"]    = ma_fn(df["close"], MA_SLOW)
    df["ma_trend"]   = ma_fn(df["close"], MA_TREND)
    df["atr"]      = _atr(df, ATR_PERIOD)
    df["rsi"]      = _rsi(df["close"], RSI_PERIOD)
    df["adx"]      = _adx(df, ADX_PERIOD)

    # Volume ortalama
    df["vol_ma"]   = df["volume"].rolling(VOLUME_MA_PERIOD).mean()

    # Stochastic RSI
    df["srsi_k"], df["srsi_d"] = _stoch_rsi(
        df["close"], SRSI_PERIOD, SRSI_K_SMOOTH, SRSI_D_SMOOTH
    )

    # MACD
    df["macd_line"], df["macd_signal"], df["macd_hist"] = _macd(
        df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL
    )

    # Supertrend
    df["st_trend"], df["st_support"] = _supertrend(df, ST_ATR_PERIOD, ST_MULTIPLIER)

    # EMA hizalama (sürekli durum: sadece kesişme anı değil)
    df["ema_bull"] = df["ma_fast"] > df["ma_slow"]

    # Kesişme olayları (durum makinesi için "silah kurma" noktası)
    prev_above     = df["ma_fast"].shift(1) > df["ma_slow"].shift(1)
    curr_above     = df["ma_fast"] > df["ma_slow"]
    df["cross_up"]   = (~prev_above) & curr_above
    df["cross_down"] = prev_above & (~curr_above)

    return df


# ---------------------------------------------------------------------------
# Sinyal üretimi (durum makinesiyle)
# ---------------------------------------------------------------------------

def get_signal(df: pd.DataFrame, state: dict | None = None) -> dict:
    """
    Son kapanan mum için giriş sinyali döndürür.

    state : durum makinesi için kalıcı sözlük — her çağrıda aynı nesne geçilmeli.
            Anahtarlar: "armed", "dipped", "oversold_seen"
            None ise ENTRY_MODE="ma_cross" davranışı kullanılır.

    Returns
    -------
    dict:
        signal     : 'buy' | 'sell' | 'none'
        price, atr, ma_fast, ma_slow, ma_trend, rsi, adx, srsi_k, srsi_d, bull
    """
    if df is None or len(df) < MA_TREND + SRSI_PERIOD + 10:
        return {"signal": "none"}

    df  = add_indicators(df)
    row = df.iloc[-2]   # son KAPANAN mum
    prv = df.iloc[-3]   # bir önceki mum

    base = {
        "signal"     : "none",
        "price"      : float(row["close"]),
        "atr"        : float(row["atr"]),
        "ma_fastest" : float(row["ma_fastest"]),
        "ma_fast"    : float(row["ma_fast"]),
        "ma_slow"    : float(row["ma_slow"]),
        "ma_trend"   : float(row["ma_trend"]),
        "rsi"        : float(row["rsi"]),
        "adx"        : float(row["adx"]),
        "srsi_k"     : float(row["srsi_k"]),
        "srsi_d"     : float(row["srsi_d"]),
        "macd_hist"  : float(row["macd_hist"]),
        "st_trend"   : int(row["st_trend"]),
        "st_support" : float(row["st_support"]),
        "bull"       : bool(row["close"] > row["ma_trend"]),
        "score"      : 0,
        "trade_risk" : RISK_LOW,
    }

    price      = base["price"]
    bull       = base["bull"]
    adx        = base["adx"]
    sk         = float(row["srsi_k"])
    sd         = float(row["srsi_d"])
    sk_prev    = float(prv["srsi_k"])
    cross_up   = bool(row["cross_up"])
    cross_down = bool(row["cross_down"])
    ema_bull   = bool(row["ema_bull"])
    vol_ok     = float(row["volume"]) >= float(row["vol_ma"]) * VOLUME_MIN_MULT

    mode = ENTRY_MODE if state is not None else "ma_cross"

    if mode == "supertrend":
        st_now = int(row["st_trend"])
        st_prv = int(prv["st_trend"])
        bull_flip = st_prv == -1 and st_now == 1   # Supertrend ayıdan boğaya döndü
        bear_flip = st_prv == 1  and st_now == -1

        if bull_flip and bull and vol_ok:
            base["signal"] = "buy"
        elif bear_flip and not bull and vol_ok:
            base["signal"] = "sell"

    elif mode == "macd_score_rider":
        mh_now  = float(row["macd_hist"])
        mh_prv  = float(prv["macd_hist"])
        rsi_v   = float(row["rsi"])
        ef_v    = float(row["ma_fast"])    # EMA21
        es_v    = float(row["ma_slow"])    # EMA55
        et_v    = float(row["ma_trend"])   # EMA200
        p_prev  = float(prv["close"])
        p_prev2 = float(df.iloc[-4]["close"]) if len(df) >= 4 else p_prev

        macd_cross_up = mh_now > 0 and mh_prv <= 0

        score = 0
        if price > et_v:                    score += 2
        if ef_v > es_v:                     score += 2
        if adx > 25:                        score += 2
        elif adx > ADX_THRESHOLD:           score += 1
        if 50 <= rsi_v <= 70:               score += 2
        elif 45 <= rsi_v <= 72:             score += 1
        if price > p_prev:                  score += 1
        if price > p_prev2:                 score += 1

        if macd_cross_up and vol_ok and score >= SCORE_MIN:
            base["signal"]     = "buy"
            base["score"]      = score
            base["trade_risk"] = RISK_HIGH if score >= SCORE_HIGH else RISK_LOW

    elif mode == "quad_ma":
        e1     = float(row["ma_fastest"])   # EMA5
        e1_prv = float(prv["ma_fastest"])
        ef     = float(row["ma_fast"])      # EMA15
        ef_prv = float(prv["ma_fast"])
        es     = float(row["ma_slow"])      # EMA50
        et     = float(row["ma_trend"])     # EMA200

        q_cross_up   = e1_prv <= ef_prv and e1 > ef
        q_cross_down = e1_prv >= ef_prv and e1 < ef
        full_bull    = e1 > ef > es > et
        full_bear    = e1 < ef < es < et

        if q_cross_up and full_bull and adx >= ADX_THRESHOLD and vol_ok:
            base["signal"] = "buy"
        elif q_cross_down and full_bear and adx >= ADX_THRESHOLD and vol_ok:
            base["signal"] = "sell"

    elif mode == "ma_cross":
        prev_above = bool(prv["ma_fast"] > prv["ma_slow"])
        curr_above = bool(row["ma_fast"] > row["ma_slow"])
        if not prev_above and curr_above and bull and adx >= ADX_THRESHOLD and vol_ok:
            base["signal"] = "buy"
        elif prev_above and not curr_above and not bull and adx >= ADX_THRESHOLD and vol_ok:
            base["signal"] = "sell"

    elif mode == "rsi_pullback":
        if state is None:
            state = {"armed": False, "dipped": False, "oversold_seen": False}
        rsi = float(row["rsi"])

        if cross_up and bull:
            state["armed"] = True
            state["dipped"] = False
        if cross_down or not ema_bull:
            state["armed"] = False
            state["dipped"] = False

        if state["armed"] and ema_bull:
            if rsi < 45:
                state["dipped"] = True
            if state["dipped"] and rsi > 52 and adx >= ADX_THRESHOLD and vol_ok:
                base["signal"] = "buy"
                state["armed"] = False
                state["dipped"] = False

    elif mode == "stochrsi":
        if state is None:
            state = {"long_armed": False, "long_seen": False,
                     "short_armed": False, "short_seen": False}
        # Geriye dönük uyumluluk (tek-dict formatı)
        if "long_armed" not in state:
            state.update({"long_armed": state.get("armed", False),
                          "long_seen": state.get("oversold_seen", False),
                          "short_armed": False, "short_seen": False})

        # === LONG: Boğa trendi sürdükçe dip ara ===
        if ema_bull and bull:
            state["long_armed"] = True
        else:
            state["long_armed"] = False
            state["long_seen"] = False

        if state["long_armed"]:
            if sk < SRSI_OVERSOLD:
                state["long_seen"] = True
            k_cross_up = sk_prev < sd and sk >= sd
            if state["long_seen"] and k_cross_up and adx >= ADX_THRESHOLD and vol_ok:
                base["signal"] = "buy"
                state["long_seen"] = False

        # === SHORT: Ayı trendi sürdükçe zirve ara ===
        if not ema_bull and not bull:
            state["short_armed"] = True
        else:
            state["short_armed"] = False
            state["short_seen"] = False

        if state["short_armed"] and base["signal"] == "none":
            if sk > SRSI_OVERBOUGHT:
                state["short_seen"] = True
            k_cross_dn = sk_prev > sd and sk <= sd
            if state["short_seen"] and k_cross_dn and adx >= ADX_THRESHOLD and vol_ok:
                base["signal"] = "sell"
                state["short_seen"] = False

    return base


def should_exit_early(df: pd.DataFrame, side: str) -> bool:
    """Trailing stop v3'te çıkışı yönetir; bu fonksiyon artık kullanılmıyor."""
    return False
