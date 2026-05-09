"""
Backtest Motoru v3 — Trailing ATR Stop + StochRSI Giriş + Volume Filtresi
=========================================================================

Çalıştırma
----------
  python backtest.py                           # simüle data (ağ gerekmez)
  python backtest.py --entry-mode ma_cross     # eski v2 karşılaştırması
  python backtest.py --entry-mode stochrsi     # yeni strateji (varsayılan)
  python backtest.py --real --limit 3000       # gerçek Binance verisi
  python backtest.py --plot                    # equity curve grafiği
"""

import argparse
import sys
import logging

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

COMMISSION = 0.001   # Binance %0.1 (BNB ile %0.075)


# ---------------------------------------------------------------------------
# Yardımcı göstergeler (backtest içinde bağımsız)
# ---------------------------------------------------------------------------

def _ema(s, p): return s.ewm(span=p, adjust=False).mean()


def _atr(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()


def _rsi(s, p=14):
    d  = s.diff()
    g  = d.clip(lower=0).ewm(span=p, adjust=False).mean()
    lo = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
    return 100 - 100 / (1 + g / lo.replace(0, 1e-9))


def _adx(df, p=14):
    h, l, c = df["high"], df["low"], df["close"]
    dmp = (h - h.shift()).clip(lower=0)
    dmn = (l.shift() - l).clip(lower=0)
    both = dmp > dmn
    dmp  = dmp.where(both, 0)
    dmn  = dmn.where(~both, 0)
    atr_ = _atr(df, p)
    dip  = 100 * dmp.ewm(span=p, adjust=False).mean() / atr_.replace(0, 1e-9)
    din  = 100 * dmn.ewm(span=p, adjust=False).mean() / atr_.replace(0, 1e-9)
    dx   = 100 * (dip - din).abs() / (dip + din).replace(0, 1e-9)
    return dx.ewm(span=p, adjust=False).mean()


def _stochrsi(s, rsi_p=14, k_p=3, d_p=3):
    rsi = _rsi(s, rsi_p)
    lo  = rsi.rolling(rsi_p).min()
    hi  = rsi.rolling(rsi_p).max()
    raw = (rsi - lo) / (hi - lo + 1e-9)
    K   = raw.rolling(k_p).mean()
    D   = K.rolling(d_p).mean()
    return K, D


def _macd(s, fast=12, slow=26, signal=9):
    """MACD çizgisi, sinyal çizgisi ve histogram döndürür."""
    line = _ema(s, fast) - _ema(s, slow)
    sig  = _ema(line, signal)
    hist = line - sig
    return line, sig, hist


def _market_score(price, ema21, ema55, ema200, adx, rsi, prev_close, prev2_close):
    """0-10 arası piyasa kalite skoru. ≥6 giriş için yeterli, ≥9 güçlü sinyal."""
    s = 0
    if price > ema200:           s += 2  # boğa rejimi
    if ema21 > ema55:            s += 2  # orta vadeli trend
    if adx > 25:                 s += 2  # güçlü trend
    elif adx > 20:               s += 1  # zayıf trend
    if 50 <= rsi <= 70:          s += 2  # pozitif momentum
    elif 45 <= rsi <= 72:        s += 1
    if price > prev_close:       s += 1  # 1-bar momentum
    if price > prev2_close:      s += 1  # 2-bar momentum
    return s


def _supertrend(df, atr_period=10, multiplier=3.0):
    """
    Supertrend göstergesi: trend (1=boğa/-1=ayı) ve destek çizgisi döndürür.
    Trend=1 iken destek = alt bant; trend=-1 iken direnç = üst bant.
    """
    atr   = _atr(df, atr_period)
    hl2   = (df["high"] + df["low"]) / 2
    close = df["close"].values
    upper = (hl2 + multiplier * atr).values
    lower = (hl2 - multiplier * atr).values
    n = len(df)

    fu = upper.copy()   # final upper band
    fl = lower.copy()   # final lower band
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

    import pandas as pd
    return (pd.Series(trend, index=df.index),
            pd.Series(support, index=df.index))


# ---------------------------------------------------------------------------
# Trailing stop yardımcısı
# ---------------------------------------------------------------------------

def _update_trail(side, entry, atr, price, sl, extreme,
                  trail_be, trail_act, trail_dist):
    """SL'yi ve extreme değerini günceller; (new_sl, new_extreme) döndürür."""
    if side == "buy":
        new_ext = max(extreme, price)
        new_sl  = sl
        move    = new_ext - entry
        if move >= trail_act * atr:
            new_sl = max(sl, new_ext - trail_dist * atr)
        elif move >= trail_be * atr:
            new_sl = max(sl, entry)
        return new_sl, new_ext
    else:
        new_ext = min(extreme, price)
        new_sl  = sl
        move    = entry - new_ext
        if move >= trail_act * atr:
            new_sl = min(sl, new_ext + trail_dist * atr)
        elif move >= trail_be * atr:
            new_sl = min(sl, entry)
        return new_sl, new_ext


# ---------------------------------------------------------------------------
# Backtest motoru
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    initial_balance: float = 10_000.0,
    fast: int = 21, slow: int = 55, trend: int = 200,
    adx_min: float = 20,
    sl_mult: float = 1.2,
    cap_mult: float = 5.5,
    trail_be: float = 2.0,
    trail_act: float = 3.0,
    trail_dist: float = 1.5,
    vol_mult: float = 1.2,
    srsi_oversold: float = 0.35,
    srsi_overbought: float = 0.65,
    exit_on_cross: bool = True,
    entry_mode: str = "ma_cross",
    risk: float = 0.01,
    long_only: bool = False,
    risk_high: float = 0.05,
    risk_low: float = 0.01,
    score_min: int = 6,
    score_high: int = 9,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal_period: int = 9,
    st_multiplier: float = 7.0,
    st_atr_period: int = 14,
) -> dict:

    # Göstergeleri hazırla
    df = df.copy()
    df["e1"]       = _ema(df["close"], 5)    # quad MA en hızlı bileşen (EMA5)
    df["ef"]       = _ema(df["close"], fast)
    df["es"]       = _ema(df["close"], slow)
    df["et"]       = _ema(df["close"], trend)
    df["atr"]      = _atr(df, 14)
    df["rsi"]      = _rsi(df["close"])
    df["adx"]      = _adx(df)
    sk, sd         = _stochrsi(df["close"])
    df["sk"]       = sk
    df["sd"]       = sd
    df["vol_ma"]   = df["volume"].rolling(20).mean()
    ml, ms, mh     = _macd(df["close"], macd_fast, macd_slow, macd_signal_period)
    df["macd_l"]   = ml
    df["macd_s"]   = ms
    df["macd_h"]   = mh
    df["rsi_ind"]  = _rsi(df["close"])
    st_trend, st_support = _supertrend(df, atr_period=st_atr_period, multiplier=st_multiplier)
    df["st_trend"]   = st_trend
    df["st_support"] = st_support
    df = df.dropna().reset_index()
    # Donchian kanalı (sadece breakout modunda kullanılır; dropna'dan sonra hesaplanır)
    # 96-bar ≈ 4 gün giriş, 48-bar ≈ 2 gün çıkış (turtle trading uyarlaması)
    df["don_high"] = df["high"].rolling(96, min_periods=1).max()
    df["don_low"]  = df["low"].rolling(96, min_periods=1).min()
    df["don_exit"] = df["low"].rolling(48, min_periods=1).min()

    balance  = initial_balance
    equity   = []
    trades   = []
    position = None

    # Durum makinesi — long ve short ayrı tutulur
    pb_long  = {"armed": False, "oversold_seen": False, "dipped": False}
    pb_short = {"armed": False, "oversold_seen": False}

    for i, row in df.iterrows():
        price = float(row["close"])
        equity.append({"time": row["timestamp"], "equity": balance})

        ef   = float(row["ef"]); es = float(row["es"])
        ef1  = float(df.loc[i - 1, "ef"]) if i > 0 else ef
        es1  = float(df.loc[i - 1, "es"]) if i > 0 else es
        ema_bull   = ef > es
        cross_up   = ef1 < es1 and ef > es
        cross_down = ef1 > es1 and ef < es

        # ----------------------------------------------------------------
        # ÇIKIŞ MANTIĞI
        # ----------------------------------------------------------------
        if position:
            atr_e = position["atr_e"]

            # Trailing stop güncelle
            new_sl, new_ext = _update_trail(
                position["side"], position["entry"], atr_e,
                price, position["sl"], position["extreme"],
                trail_be, trail_act, trail_dist,
            )
            position["sl"]      = new_sl
            position["extreme"] = new_ext

            sd_dir = 1 if position["side"] == "buy" else -1
            cap_hit   = (sd_dir ==  1 and price >= position["cap_tp"]) or \
                        (sd_dir == -1 and price <= position["cap_tp"])
            sl_hit    = (sd_dir ==  1 and price <= position["sl"]) or \
                        (sd_dir == -1 and price >= position["sl"])
            # supertrend: destek çizgisini SL olarak güncelle, flip'te çık
            if entry_mode == "supertrend":
                sup_now = float(row["st_support"])
                if position["side"] == "buy":
                    position["sl"] = max(position["sl"], sup_now)
                else:
                    position["sl"] = min(position["sl"], sup_now)
                st_now = int(row["st_trend"])
                st_prv = int(df.loc[i - 1, "st_trend"])
                st_flip_bear = st_prv == 1 and st_now == -1
                st_flip_bull = st_prv == -1 and st_now == 1
                st_exit = (position["side"] == "buy" and st_flip_bear) or \
                          (position["side"] == "sell" and st_flip_bull)
                if st_exit:
                    sd_dir = 1 if position["side"] == "buy" else -1
                    pnl    = (price - position["entry"]) * position["qty"] * sd_dir
                    fee    = price * position["qty"] * COMMISSION
                    net    = pnl - fee
                    balance += net
                    trades.append({
                        "entry_time": position["entry_time"], "exit_time": row["timestamp"],
                        "side": position["side"], "entry": position["entry"],
                        "exit": price, "qty": position["qty"],
                        "pnl_gross": round(pnl, 4), "pnl_net": round(net, 4),
                        "reason": "st_flip",
                        "peak_move": round(abs(position["extreme"] - position["entry"]), 4),
                    })
                    position = None
                    continue

            # breakout: Donchian düşüğün altına inince çıkış
            if entry_mode == "breakout":
                don_exit_now = float(row["don_exit"])
                don_exit_brk = (position["side"] == "buy" and price < don_exit_now) or \
                               (position["side"] == "sell" and price > don_exit_now)
                if don_exit_brk:
                    sd_dir = 1 if position["side"] == "buy" else -1
                    pnl    = (price - position["entry"]) * position["qty"] * sd_dir
                    fee    = price * position["qty"] * COMMISSION
                    net    = pnl - fee
                    balance += net
                    trades.append({
                        "entry_time": position["entry_time"], "exit_time": row["timestamp"],
                        "side": position["side"], "entry": position["entry"],
                        "exit": price, "qty": position["qty"],
                        "pnl_gross": round(pnl, 4), "pnl_net": round(net, 4),
                        "reason": "don_exit",
                        "peak_move": round(abs(position["extreme"] - position["entry"]), 4),
                    })
                    position = None
                    continue

            # quad_ma modunda EMA5/EMA_fast çaprazı; diğer modlarda EMA21/EMA55 çaprazı
            if entry_mode == "quad_ma":
                e1_now = float(row["e1"])
                e1_prv = float(df.loc[i - 1, "e1"])
                ef_now = float(row["ef"]); ef_prv = float(df.loc[i - 1, "ef"])
                _cross_down = e1_prv > ef_prv and e1_now <= ef_now
                _cross_up   = e1_prv < ef_prv and e1_now >= ef_now
            else:
                _cross_down = cross_down
                _cross_up   = cross_up
            cross_hit = exit_on_cross and (
                (position["side"] == "buy"  and _cross_down) or
                (position["side"] == "sell" and _cross_up)
            )

            if cap_hit or sl_hit or cross_hit:
                if cap_hit:   reason = "cap_tp"
                elif sl_hit:  reason = "trail_sl"
                else:         reason = "cross_exit"
                pnl  = (price - position["entry"]) * position["qty"] * sd_dir
                fee  = price * position["qty"] * COMMISSION
                net  = pnl - fee
                balance += net
                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time" : row["timestamp"],
                    "side"      : position["side"],
                    "entry"     : position["entry"],
                    "exit"      : price,
                    "qty"       : position["qty"],
                    "pnl_gross" : round(pnl, 4),
                    "pnl_net"   : round(net, 4),
                    "reason"    : reason,
                    "peak_move" : round(abs(position["extreme"] - position["entry"]), 4),
                })
                position = None
            continue

        # ----------------------------------------------------------------
        # GİRİŞ MANTIĞI
        # ----------------------------------------------------------------
        if i < 5:
            continue

        bull = price > float(row["et"])
        adx  = float(row["adx"])
        sk_v = float(row["sk"]); sd_v = float(row["sd"])
        sk1  = float(df.loc[i - 1, "sk"])
        vol  = float(row["volume"]); vma = float(row["vol_ma"])
        atr  = float(row["atr"])

        vol_ok = vol >= vol_mult * vma if vol_mult > 0 else True

        sig = None

        # --- MA Cross ---
        if entry_mode == "ma_cross":
            if cross_up and bull and adx >= adx_min and vol_ok:
                sig = "buy"
            elif not long_only and cross_down and not bull and adx >= adx_min and vol_ok:
                sig = "sell"

        # --- RSI Pullback ---
        elif entry_mode == "rsi_pullback":
            rsi = float(row["rsi"])
            if cross_up and bull:
                pb_long["armed"] = True; pb_long["dipped"] = False
            if cross_down or not ema_bull or not bull:
                pb_long["armed"] = False; pb_long["dipped"] = False
            if pb_long["armed"] and ema_bull:
                if rsi < 45:
                    pb_long["dipped"] = True
                if pb_long["dipped"] and rsi > 55 and adx >= adx_min and vol_ok:
                    sig = "buy"
                    pb_long["armed"] = False; pb_long["dipped"] = False

        # --- StochRSI: her cross_up için bir giriş, sonra sıfırla ---
        elif entry_mode == "stochrsi":
            # LONG
            if cross_up and bull:
                pb_long["armed"] = True; pb_long["oversold_seen"] = False
            if cross_down or not bull:
                pb_long["armed"] = False; pb_long["oversold_seen"] = False

            if pb_long["armed"] and ema_bull:
                if sk_v < srsi_oversold:
                    pb_long["oversold_seen"] = True
                k_cross_up = sk_v > sd_v and sk1 <= sd_v
                if pb_long["oversold_seen"] and k_cross_up and adx >= adx_min and vol_ok:
                    sig = "buy"
                    pb_long["armed"] = False; pb_long["oversold_seen"] = False

            # SHORT
            if not long_only and sig is None:
                if cross_down and not bull:
                    pb_short["armed"] = True; pb_short["oversold_seen"] = False
                if cross_up or bull:
                    pb_short["armed"] = False; pb_short["oversold_seen"] = False

                if pb_short["armed"] and not ema_bull:
                    if sk_v > srsi_overbought:
                        pb_short["oversold_seen"] = True
                    k_cross_dn = sk_v < sd_v and sk1 >= sd_v
                    if pb_short["oversold_seen"] and k_cross_dn and adx >= adx_min and vol_ok:
                        sig = "sell"
                        pb_short["armed"] = False; pb_short["oversold_seen"] = False

        # --- Dörtlü MA (quad_ma): EMA5/15/50/200 hizalaması ---
        elif entry_mode == "quad_ma":
            e1   = float(row["e1"])    # EMA5
            e1_1 = float(df.loc[i - 1, "e1"])
            ef_v = float(row["ef"])    # EMA fast (15 veya 21)
            ef_1 = float(df.loc[i - 1, "ef"])
            # Long: EMA5 EMA_fast'ı yukarı kesiyor VE EMA_fast > EMA_slow > EMA_trend (tam boğa)
            q_cross_up   = e1_1 <= ef_1 and e1 > ef_v
            q_cross_down = e1_1 >= ef_1 and e1 < ef_v
            full_bull = e1 > ef_v > float(row["es"]) > float(row["et"])
            full_bear = e1 < ef_v < float(row["es"]) < float(row["et"])

            if q_cross_up and full_bull and adx >= adx_min and vol_ok:
                sig = "buy"
            elif not long_only and q_cross_down and full_bear and adx >= adx_min and vol_ok:
                sig = "sell"

        # --- Donchian Kanal Kırılımı (breakout) ---
        elif entry_mode == "breakout":
            don_high = float(row["don_high"])   # N-bar yüksek
            don_low  = float(row["don_low"])    # N-bar düşük
            don_exit = float(row["don_exit"])   # çıkış için M-bar düşük
            don_h1   = float(df.loc[i - 1, "don_high"])
            don_l1   = float(df.loc[i - 1, "don_low"])
            # Fiyat N-bar yüksekliğini kırdı → giriş
            if price > don_h1 and bull and adx >= adx_min and vol_ok:
                sig = "buy"
            elif not long_only and price < don_l1 and not bull and adx >= adx_min and vol_ok:
                sig = "sell"

        # --- Supertrend: trend flip + EMA200 rejim, destek çizgisi = SL ---
        elif entry_mode == "supertrend":
            st_now = int(row["st_trend"])
            st_prv = int(df.loc[i - 1, "st_trend"])
            flip_bull = st_prv == -1 and st_now == 1
            flip_bear = st_prv == 1  and st_now == -1

            if flip_bull and bull and vol_ok:
                sig = "buy"
            elif not long_only and flip_bear and not bull and vol_ok:
                sig = "sell"

        # --- MACD Score Rider: MACD histogram crossover + piyasa skoru ---
        elif entry_mode == "macd_score_rider":
            mh_now  = float(row["macd_h"])
            mh_prv  = float(df.loc[i - 1, "macd_h"])
            rsi_v   = float(row["rsi_ind"])
            ef_v    = float(row["ef"])    # EMA21
            es_v    = float(row["es"])    # EMA55
            et_v    = float(row["et"])    # EMA200
            p_prev  = float(df.loc[i - 1, "close"])
            p_prev2 = float(df.loc[i - 2, "close"]) if i >= 2 else p_prev

            macd_cross_up = mh_now > 0 and mh_prv <= 0
            score = _market_score(price, ef_v, es_v, et_v, adx, rsi_v, p_prev, p_prev2)

            if macd_cross_up and vol_ok and score >= score_min:
                sig = "buy"
                _trade_risk = risk_high if score >= score_high else risk_low

        if sig is None:
            continue

        # Supertrend: SL = destek çizgisi (ATR SL değil)
        if entry_mode == "supertrend":
            sup = float(row["st_support"])
            sl      = sup if sig == "buy" else (2 * price - sup)  # sell için simetrik
            cap_tp  = price + cap_mult * atr if sig == "buy" else price - cap_mult * atr
        else:
            sl      = price - sl_mult * atr if sig == "buy" else price + sl_mult * atr
            cap_tp  = price + cap_mult * atr if sig == "buy" else price - cap_mult * atr

        sl_dist = max(abs(price - sl), price * 0.001)  # min %0.1 koruma
        if entry_mode == "macd_score_rider":
            qty = (balance * _trade_risk / sl_dist) if sl_dist > 0 else 0
        else:
            qty = (balance * risk / sl_dist) if sl_dist > 0 else 0

        if qty <= 0:
            continue

        fee = price * qty * COMMISSION
        balance -= fee
        position = {
            "side"      : sig,
            "entry"     : price,
            "qty"       : qty,
            "sl"        : sl,
            "cap_tp"    : cap_tp,
            "atr_e"     : atr,
            "extreme"   : price,
            "entry_time": row["timestamp"],
        }

    # Özet
    wins  = [t for t in trades if t["pnl_net"] > 0]
    loss  = [t for t in trades if t["pnl_net"] <= 0]
    total = sum(t["pnl_net"] for t in trades)
    pf    = abs(sum(t["pnl_net"] for t in wins) /
                sum(t["pnl_net"] for t in loss)) \
            if loss and sum(t["pnl_net"] for t in loss) != 0 else float("inf")

    return {
        "initial_balance"  : initial_balance,
        "final_balance"    : round(balance, 4),
        "total_pnl"        : round(total, 4),
        "total_return_pct" : round(total / initial_balance * 100, 2),
        "n_trades"         : len(trades),
        "n_wins"           : len(wins),
        "n_losses"         : len(loss),
        "win_rate_pct"     : round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_win"          : round(sum(t["pnl_net"] for t in wins) / len(wins), 4) if wins else 0,
        "avg_loss"         : round(sum(t["pnl_net"] for t in loss) / len(loss), 4) if loss else 0,
        "profit_factor"    : round(pf, 2),
        "trades"           : trades,
        "equity"           : equity,
    }


# ---------------------------------------------------------------------------
# Rapor & görselleştirme
# ---------------------------------------------------------------------------

def print_report(r: dict, entry_mode: str = "ma_cross"):
    from config import MA_FAST, MA_SLOW, MA_TREND, ATR_SL_MULT, ATR_CAP_MULT, RISK_PER_TRADE, EXIT_ON_CROSS
    sep = "─" * 58
    cross_tag = " +CrossExit" if EXIT_ON_CROSS else ""
    print(sep)
    print(f"  Strateji v3: EMA{MA_FAST}/EMA{MA_SLOW}  Rejim:EMA{MA_TREND}  [{entry_mode}]{cross_tag}")
    print(f"  SL={ATR_SL_MULT}×ATR  CapTP={ATR_CAP_MULT}×ATR  Trailing+Cross  Risk={RISK_PER_TRADE*100:.0f}%")
    print(sep)
    print(f"  Başlangıç bakiye : {r['initial_balance']:>12,.2f} USDT")
    print(f"  Son bakiye       : {r['final_balance']:>12,.2f} USDT")
    print(f"  Toplam PnL (net) : {r['total_pnl']:>+12,.4f} USDT  ({r['total_return_pct']:+.2f}%)")
    print(sep)
    print(f"  İşlem sayısı     : {r['n_trades']}")
    print(f"  Kazanan / Kaybeden: {r['n_wins']} / {r['n_losses']}")
    print(f"  Kazanma oranı    : {r['win_rate_pct']:.1f}%")
    print(f"  Ort. kazanç      : {r['avg_win']:>+.4f} USDT")
    print(f"  Ort. kayıp       : {r['avg_loss']:>+.4f} USDT")
    print(f"  Profit factor    : {r['profit_factor']:.2f}")
    print(sep)


def plot_equity(r: dict):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib")
        return

    eq  = pd.DataFrame(r["equity"]).set_index("time")
    pnl = [t["pnl_net"] for t in r["trades"]]
    tms = [t["exit_time"] for t in r["trades"]]
    clr = ["green" if p > 0 else "red" for p in pnl]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(eq.index, eq["equity"], lw=1.5, color="steelblue")
    axes[0].set_title("Equity Curve — EMA21/55 + Trailing ATR Stop + StochRSI")
    axes[0].set_ylabel("Bakiye (USDT)"); axes[0].grid(alpha=0.3)

    axes[1].bar(tms, pnl, color=clr, width=0.6)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("İşlem PnL"); axes[1].set_ylabel("PnL (USDT)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_equity.png", dpi=150)
    print("Grafik kaydedildi → backtest_equity.png")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Strateji v3 Backtest")
    parser.add_argument("--real",         action="store_true")
    parser.add_argument("--limit",        type=int, default=2000)
    parser.add_argument("--balance",      type=float, default=10_000.0)
    parser.add_argument("--long-only",    action="store_true")
    parser.add_argument("--plot",         action="store_true")
    parser.add_argument("--entry-mode",   default="supertrend",
                        choices=["ma_cross", "rsi_pullback", "stochrsi", "quad_ma", "breakout", "macd_score_rider", "supertrend"])
    parser.add_argument("--no-cross-exit", action="store_true",
                        help="MA tersine dönüşünde çıkışı devre dışı bırak")
    parser.add_argument("--compare",      action="store_true",
                        help="Tüm giriş modlarını karşılaştır")
    args = parser.parse_args()

    if args.real:
        import exchange_client as ex
        log.info("%d mum çekiliyor…", args.limit)
        df = ex.fetch_ohlcv(limit=args.limit)
        if df is None or df.empty:
            print("HATA: Veri alınamadı.")
            sys.exit(1)
    else:
        from historical_data import generate_btc_2024
        log.info("2024 BTC simüle verisi kullanılıyor…")
        df = generate_btc_2024(seed=0)

    bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    log.info("%d mum  |  Al-Tut: %+.1f%%", len(df), bh)

    use_cross = not args.no_cross_exit

    if args.compare:
        # Her mod için optimize edilmiş parametreler
        mode_params = {
            "supertrend"      : dict(fast=21, slow=55, sl_mult=1.5, cap_mult=15.0,
                                     trail_be=2.0, trail_act=4.0, trail_dist=2.0,
                                     exit_on_cross=False, st_multiplier=7.0, st_atr_period=14),
            "macd_score_rider": dict(fast=21, slow=55, sl_mult=2.0, cap_mult=20.0,
                                     trail_be=2.5, trail_act=4.5, trail_dist=3.0,
                                     exit_on_cross=False, risk_high=0.05, risk_low=0.01,
                                     score_min=6, score_high=9),
            "ma_cross"        : dict(fast=21, slow=55, sl_mult=1.2, cap_mult=5.5,
                                     trail_be=2.0, trail_act=3.0, trail_dist=1.5, exit_on_cross=True),
            "quad_ma"         : dict(fast=15, slow=50, sl_mult=1.2, cap_mult=5.5,
                                     trail_be=2.0, trail_act=3.0, trail_dist=1.5, exit_on_cross=True),
            "breakout"        : dict(fast=21, slow=55, sl_mult=1.2, cap_mult=8.0,
                                     trail_be=1.5, trail_act=2.5, trail_dist=2.0, exit_on_cross=True),
            "rsi_pullback"    : dict(fast=21, slow=55, sl_mult=1.5, cap_mult=5.5,
                                     trail_be=2.0, trail_act=3.0, trail_dist=1.5, exit_on_cross=True),
            "stochrsi"        : dict(fast=21, slow=55, sl_mult=1.5, cap_mult=5.5,
                                     trail_be=2.0, trail_act=3.0, trail_dist=1.5, exit_on_cross=True),
        }
        print(f"\n{'Mod':<20} {'Getiri':>8} {'İşlem':>6} {'KazO%':>7} {'PF':>6}  {'Ort.Kazan':>10}  {'Ort.Kayıp':>10}")
        print("─" * 75)
        for mode in ["supertrend", "macd_score_rider", "ma_cross", "quad_ma", "breakout", "rsi_pullback", "stochrsi"]:
            kw = dict(mode_params.get(mode, {}))
            ec = kw.pop("exit_on_cross", use_cross)
            r = run_backtest(df, initial_balance=args.balance,
                             long_only=args.long_only, entry_mode=mode,
                             exit_on_cross=ec, **kw)
            print(f"{mode:<20} {r['total_return_pct']:>+7.2f}% {r['n_trades']:>6} "
                  f"{r['win_rate_pct']:>6.1f}% {r['profit_factor']:>6.2f}  "
                  f"{r['avg_win']:>+10.2f}  {r['avg_loss']:>+10.2f}")
        print(f"{'Al-Tut':<20} {bh:>+7.2f}%")
    else:
        result = run_backtest(df, initial_balance=args.balance,
                              long_only=args.long_only, entry_mode=args.entry_mode,
                              exit_on_cross=use_cross)
        print_report(result, args.entry_mode)
        print(f"  Al-Tut getirisi  : {bh:>+12.2f}%")
        print("─" * 58)

        if args.plot:
            plot_equity(result)


if __name__ == "__main__":
    main()
