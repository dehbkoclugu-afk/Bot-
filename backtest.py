"""
Backtester — simulate the MA crossover strategy on historical data.

Usage
-----
  python backtest.py                     # default: BTC/USDT, last 500 candles
  python backtest.py --symbol ETH/USDT --limit 1000
  python backtest.py --plot              # show equity curve (requires matplotlib)
"""

import argparse
import logging
import sys
import pandas as pd

import exchange_client as ex
from strategy     import add_indicators
from risk_manager import calculate_levels, calculate_position_size, check_exit
from config       import MA_FAST, MA_SLOW, ATR_SL_MULT, ATR_TP_MULT, RISK_PER_TRADE

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_backtest(df: pd.DataFrame, initial_balance: float = 10_000.0) -> dict:
    df = add_indicators(df)
    df = df.dropna().reset_index()

    balance   = initial_balance
    equity    = []
    trades    = []
    position  = None

    for i, row in df.iterrows():
        price = float(row["close"])
        equity.append({"time": row["timestamp"], "equity": balance})

        # --- Check exit for open position ---
        if position:
            result = check_exit(
                position["side"], position["entry"],
                price, position["take_profit"], position["stop_loss"],
            )
            if result != "hold":
                close_side = 1 if position["side"] == "buy" else -1
                pnl = (price - position["entry"]) * position["qty"] * close_side
                balance += pnl
                trades.append({
                    "entry_time" : position["entry_time"],
                    "exit_time"  : row["timestamp"],
                    "side"       : position["side"],
                    "entry"      : position["entry"],
                    "exit"       : price,
                    "qty"        : position["qty"],
                    "pnl"        : round(pnl, 4),
                    "reason"     : result,
                })
                position = None
            else:
                continue

        # --- Check for new signal (second-to-last candle logic: use current row) ---
        if i < 2:
            continue

        cross = int(row["cross"])
        if cross == 0:
            continue

        side   = "buy" if cross == 1 else "sell"
        atr    = float(row["atr"])
        levels = calculate_levels(price, atr, side)
        tp     = levels["take_profit"]
        sl     = levels["stop_loss"]
        qty    = calculate_position_size(balance, price, sl)

        if qty <= 0:
            continue

        position = {
            "side"        : side,
            "entry"       : price,
            "qty"         : qty,
            "take_profit" : tp,
            "stop_loss"   : sl,
            "entry_time"  : row["timestamp"],
        }

    # --- Summary ---
    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl     = sum(t["pnl"] for t in trades)
    win_rate      = len(wins) / len(trades) * 100 if trades else 0
    avg_win       = sum(t["pnl"] for t in wins)   / len(wins)   if wins   else 0
    avg_loss      = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    profit_factor = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) \
                    if losses and sum(t["pnl"] for t in losses) != 0 else float("inf")

    result = {
        "initial_balance" : initial_balance,
        "final_balance"   : round(balance, 4),
        "total_pnl"       : round(total_pnl, 4),
        "total_return_pct": round(total_pnl / initial_balance * 100, 2),
        "n_trades"        : len(trades),
        "n_wins"          : len(wins),
        "n_losses"        : len(losses),
        "win_rate_pct"    : round(win_rate, 1),
        "avg_win"         : round(avg_win,  4),
        "avg_loss"        : round(avg_loss, 4),
        "profit_factor"   : round(profit_factor, 2),
        "trades"          : trades,
        "equity"          : equity,
    }
    return result


def print_report(r: dict):
    sep = "─" * 48
    print(sep)
    print(f"  MA Crossover Backtest — MA{MA_FAST} / MA{MA_SLOW}")
    print(f"  SL={ATR_SL_MULT}×ATR  TP={ATR_TP_MULT}×ATR  Risk={RISK_PER_TRADE*100:.0f}%")
    print(sep)
    print(f"  Initial balance : {r['initial_balance']:>10,.2f} USDT")
    print(f"  Final balance   : {r['final_balance']:>10,.2f} USDT")
    print(f"  Total PnL       : {r['total_pnl']:>+10,.4f} USDT  ({r['total_return_pct']:+.2f}%)")
    print(sep)
    print(f"  Trades          : {r['n_trades']}")
    print(f"  Win / Loss      : {r['n_wins']} / {r['n_losses']}")
    print(f"  Win rate        : {r['win_rate_pct']:.1f}%")
    print(f"  Avg win         : {r['avg_win']:>+.4f} USDT")
    print(f"  Avg loss        : {r['avg_loss']:>+.4f} USDT")
    print(f"  Profit factor   : {r['profit_factor']:.2f}")
    print(sep)


def plot_equity(r: dict):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — run: pip install matplotlib")
        return

    eq = pd.DataFrame(r["equity"])
    eq.set_index("time", inplace=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(eq.index, eq["equity"], label="Equity", color="steelblue", lw=1.5)
    axes[0].set_title(f"Equity Curve — MA{MA_FAST}/MA{MA_SLOW} Crossover")
    axes[0].set_ylabel("Balance (USDT)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    pnls  = [t["pnl"] for t in r["trades"]]
    times = [t["exit_time"] for t in r["trades"]]
    colors = ["green" if p > 0 else "red" for p in pnls]
    axes[1].bar(times, pnls, color=colors, width=0.6)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Trade PnL")
    axes[1].set_ylabel("PnL (USDT)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest_equity.png", dpi=150)
    print("Chart saved → backtest_equity.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Backtest MA crossover bot")
    parser.add_argument("--symbol", default=None,  help="e.g. ETH/USDT")
    parser.add_argument("--limit",  type=int, default=500)
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--plot",   action="store_true")
    args = parser.parse_args()

    if args.symbol:
        import config
        config.SYMBOL = args.symbol

    log.info("Fetching %d candles…", args.limit)
    df = ex.fetch_ohlcv(limit=args.limit)
    if df is None or df.empty:
        print("ERROR: could not fetch data. Check your API keys / network.")
        sys.exit(1)

    log.info("Running backtest on %d candles…", len(df))
    result = run_backtest(df, initial_balance=args.balance)
    print_report(result)

    if args.plot:
        plot_equity(result)


if __name__ == "__main__":
    main()
