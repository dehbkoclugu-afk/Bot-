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

        if sig is None:
            continue

        sl     = price - sl_mult * atr if sig == "buy" else price + sl_mult * atr
        cap_tp = price + cap_mult * atr if sig == "buy" else price - cap_mult * atr
        sl_dist = abs(price - sl)
        qty    = (balance * risk / sl_dist) if sl_dist > 0 else 0

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
    parser.add_argument("--entry-mode",   default="ma_cross",
                        choices=["ma_cross", "rsi_pullback", "stochrsi", "quad_ma", "breakout"])
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
            "ma_cross"   : dict(fast=21, slow=55, sl_mult=1.2, cap_mult=5.5, trail_be=2.0, trail_act=3.0, trail_dist=1.5),
            "quad_ma"    : dict(fast=15, slow=50, sl_mult=1.2, cap_mult=5.5, trail_be=2.0, trail_act=3.0, trail_dist=1.5),
            "breakout"   : dict(fast=21, slow=55, sl_mult=1.2, cap_mult=8.0, trail_be=1.5, trail_act=2.5, trail_dist=2.0),
            "rsi_pullback": dict(fast=21, slow=55, sl_mult=1.5, cap_mult=5.5, trail_be=2.0, trail_act=3.0, trail_dist=1.5),
            "stochrsi"   : dict(fast=21, slow=55, sl_mult=1.5, cap_mult=5.5, trail_be=2.0, trail_act=3.0, trail_dist=1.5),
        }
        print(f"\n{'Mod':<15} {'Getiri':>8} {'İşlem':>6} {'KazO%':>7} {'PF':>6}  {'Ort.Kazan':>10}  {'Ort.Kayıp':>10}")
        print("─" * 68)
        for mode in ["ma_cross", "quad_ma", "breakout", "rsi_pullback", "stochrsi"]:
            kw = mode_params.get(mode, {})
            r = run_backtest(df, initial_balance=args.balance,
                             long_only=args.long_only, entry_mode=mode,
                             exit_on_cross=use_cross, **kw)
            print(f"{mode:<15} {r['total_return_pct']:>+7.2f}% {r['n_trades']:>6} "
                  f"{r['win_rate_pct']:>6.1f}% {r['profit_factor']:>6.2f}  "
                  f"{r['avg_win']:>+10.2f}  {r['avg_loss']:>+10.2f}")
        print(f"{'Al-Tut':<15} {bh:>+7.2f}%")
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
