"""
Backtest Motoru — EMA 21/55 + 200 Rejim Filtresi
=================================================

Çalıştırma
----------
  python backtest.py                        # simüle data (ağ gerekmez)
  python backtest.py --real                 # gerçek Binance verisi (API gerekir)
  python backtest.py --real --limit 2000    # daha fazla geçmiş
  python backtest.py --plot                 # equity curve grafiği
"""

import argparse
import sys
import logging

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# Tek işlem maliyet katsayısı
# ---------------------------------------------------------------------------
COMMISSION = 0.001   # Binance spot %0.1 (BNB ile %0.075)


# ---------------------------------------------------------------------------
# Yardımcı göstergeler (backtest'te strategy.py'den bağımsız çalışır)
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


# ---------------------------------------------------------------------------
# Backtest motoru
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame,
                 initial_balance: float = 10_000.0,
                 fast: int = 21, slow: int = 55, trend: int = 200,
                 adx_min: float = 20,
                 rsi_long_min: float = 38, rsi_long_max: float = 72,
                 rsi_short_min: float = 28, rsi_short_max: float = 62,
                 sl_mult: float = 1.5, tp_mult: float = 3.5,
                 risk: float = 0.01,
                 long_only: bool = False,
                 exit_on_cross: bool = True) -> dict:

    df = df.copy()
    df["ef"]    = _ema(df["close"], fast)
    df["es"]    = _ema(df["close"], slow)
    df["et"]    = _ema(df["close"], trend)
    df["atr"]   = _atr(df, 14)
    df["rsi"]   = _rsi(df["close"])
    df["adx"]   = _adx(df)
    df = df.dropna().reset_index()

    balance  = initial_balance
    equity   = []
    trades   = []
    position = None

    for i, row in df.iterrows():
        price = float(row["close"])
        equity.append({"time": row["timestamp"], "equity": balance})

        if position:
            side_dir = 1 if position["side"] == "buy" else -1
            tp_hit = (side_dir ==  1 and price >= position["tp"]) or \
                     (side_dir == -1 and price <= position["tp"])
            sl_hit = (side_dir ==  1 and price <= position["sl"]) or \
                     (side_dir == -1 and price >= position["sl"])

            # Erken çıkış: MA tersine döndüyse
            ef  = float(row["ef"]); es = float(row["es"])
            ef1 = float(df.loc[i - 1, "ef"]); es1 = float(df.loc[i - 1, "es"])
            cross_exit = exit_on_cross and (
                (side_dir ==  1 and ef1 > es1 and ef < es) or
                (side_dir == -1 and ef1 < es1 and ef > es)
            )

            if tp_hit or sl_hit or cross_exit:
                pnl = (price - position["entry"]) * position["qty"] * side_dir
                fee = price * position["qty"] * COMMISSION
                net = pnl - fee
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
                    "reason"    : "tp" if tp_hit else ("sl" if sl_hit else "cross"),
                })
                position = None
            continue

        # Sinyal üret
        if i < 3:
            continue

        ef   = float(row["ef"]); es = float(row["es"])
        ef1  = float(df.loc[i - 1, "ef"]); es1 = float(df.loc[i - 1, "es"])
        bull = price > float(row["et"])
        rsi  = float(row["rsi"])
        adx  = float(row["adx"])

        if adx < adx_min:
            continue

        sig = None
        if ef1 < es1 and ef > es and bull and rsi_long_min < rsi < rsi_long_max:
            sig = "buy"
        elif not long_only and ef1 > es1 and ef < es and not bull \
                and rsi_short_min < (100 - rsi) < rsi_short_max:
            sig = "sell"

        if sig is None:
            continue

        atr = float(row["atr"])
        sl  = price - sl_mult * atr if sig == "buy" else price + sl_mult * atr
        tp  = price + tp_mult * atr if sig == "buy" else price - tp_mult * atr
        sl_dist = abs(price - sl)
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
            "tp"        : tp,
            "entry_time": row["timestamp"],
        }

    # Özet
    wins   = [t for t in trades if t["pnl_net"] > 0]
    losses = [t for t in trades if t["pnl_net"] <= 0]
    total  = sum(t["pnl_net"] for t in trades)
    pf     = abs(sum(t["pnl_net"] for t in wins) /
                 sum(t["pnl_net"] for t in losses)) \
             if losses and sum(t["pnl_net"] for t in losses) != 0 else float("inf")

    return {
        "initial_balance"  : initial_balance,
        "final_balance"    : round(balance, 4),
        "total_pnl"        : round(total, 4),
        "total_return_pct" : round(total / initial_balance * 100, 2),
        "n_trades"         : len(trades),
        "n_wins"           : len(wins),
        "n_losses"         : len(losses),
        "win_rate_pct"     : round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_win"          : round(sum(t["pnl_net"] for t in wins) / len(wins), 4) if wins else 0,
        "avg_loss"         : round(sum(t["pnl_net"] for t in losses) / len(losses), 4) if losses else 0,
        "profit_factor"    : round(pf, 2),
        "trades"           : trades,
        "equity"           : equity,
    }


# ---------------------------------------------------------------------------
# Rapor & görselleştirme
# ---------------------------------------------------------------------------

def print_report(r: dict):
    sep = "─" * 52
    from config import MA_FAST, MA_SLOW, MA_TREND, ATR_SL_MULT, ATR_TP_MULT, RISK_PER_TRADE
    print(sep)
    print(f"  Strateji: EMA{MA_FAST}/EMA{MA_SLOW}  Rejim: EMA{MA_TREND}")
    print(f"  SL={ATR_SL_MULT}×ATR  TP={ATR_TP_MULT}×ATR  Risk={RISK_PER_TRADE*100:.0f}%  Komisyon=%0.1")
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
        print("matplotlib yüklü değil: pip install matplotlib")
        return

    eq  = pd.DataFrame(r["equity"]).set_index("time")
    pnl = [t["pnl_net"] for t in r["trades"]]
    tms = [t["exit_time"] for t in r["trades"]]
    clr = ["green" if p > 0 else "red" for p in pnl]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(eq.index, eq["equity"], lw=1.5, color="steelblue")
    axes[0].set_title("Equity Curve — EMA21/55 + Rejim Filtresi")
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
    parser = argparse.ArgumentParser(description="EMA 21/55 Strateji Backtest")
    parser.add_argument("--real",      action="store_true", help="Gerçek Binance verisi (API gerekir)")
    parser.add_argument("--limit",     type=int, default=2000)
    parser.add_argument("--balance",   type=float, default=10_000.0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--plot",      action="store_true")
    args = parser.parse_args()

    if args.real:
        import exchange_client as ex
        log.info("%d mum çekiliyor…", args.limit)
        df = ex.fetch_ohlcv(limit=args.limit)
        if df is None or df.empty:
            print("HATA: Veri alınamadı. API anahtarlarını ve ağ bağlantısını kontrol edin.")
            sys.exit(1)
    else:
        from historical_data import generate_btc_2024
        log.info("Simüle edilmiş 2024 BTC verisi kullanılıyor…")
        df = generate_btc_2024(seed=0)

    log.info("%d mum üzerinde backtest çalışıyor…", len(df))
    bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    result = run_backtest(df, initial_balance=args.balance, long_only=args.long_only)
    print_report(result)
    print(f"  Al-Tut getirisi  : {bh:>+12.2f}%")
    winner = "BOT" if result["total_return_pct"] > bh else "AL-TUT"
    print(f"  Kazanan          : {winner}")
    print("─" * 52)

    if args.plot:
        plot_equity(result)


if __name__ == "__main__":
    main()
