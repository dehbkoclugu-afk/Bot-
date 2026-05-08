"""
2024 BTC/USDT Gerçekçi Backtest
================================
Gerçek pivot noktalarından üretilen 2024 BTC fiyat serisi üzerinde
MA5/MA12 crossover stratejisini (ADX filtreli) test eder.

Çalıştır:
  python real_backtest.py
  python real_backtest.py --plot
"""

import argparse
from historical_data import generate_btc_2024
from backtest        import run_backtest, print_report, plot_equity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--balance", type=float, default=10_000.0)
    args = parser.parse_args()

    print("\n2024 BTC/USDT fiyat verisi hazırlanıyor (8 760 saatlik mum)…")
    df = generate_btc_2024(seed=0)
    print(f"Dönem  : {df.index[0].strftime('%d.%m.%Y')} → {df.index[-1].strftime('%d.%m.%Y')}")
    print(f"Fiyat  : {df['close'].min():>10,.0f} → {df['close'].max():>10,.0f} USDT")
    print(f"Değişim: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:>+.1f}%  (al-tut)\n")

    result = run_backtest(df, initial_balance=args.balance)
    print_report(result)

    # Buy & Hold karşılaştırması
    bh_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    print(f"  Al-Tut karşılaştırması : {bh_return:>+.2f}%")
    print(f"  Bot getirisi           : {result['total_return_pct']:>+.2f}%")
    winner = "BOT" if result["total_return_pct"] > bh_return else "AL-TUT"
    print(f"  Kazanan                : {winner}")
    print("─" * 48)

    # Aylık bazda işlemler
    trades = result["trades"]
    if trades:
        print(f"\n{'#':>3}  {'Yön':<5}  {'Giriş':>10}  {'Çıkış':>10}  {'PnL$':>9}  {'%':>6}  Neden")
        print("─" * 68)
        for i, t in enumerate(trades, 1):
            pnl_pct = (t["exit"] - t["entry"]) / t["entry"] * 100 * (1 if t["side"]=="buy" else -1)
            print(
                f"{i:>3}  {t['side'].upper():<5}  "
                f"{t['entry']:>10,.0f}  {t['exit']:>10,.0f}  "
                f"{t['pnl']:>+9.2f}  {pnl_pct:>+5.1f}%  {t['reason']}"
            )

    if args.plot:
        plot_equity(result)


if __name__ == "__main__":
    main()
