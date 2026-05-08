"""
Offline Backtest — Simulated BTC/USDT price data (no internet needed).

Üretilen fiyat serisi gerçekçi bir kripto hareketi modelliyor:
- GBM (Geometric Brownian Motion) + momentum bias
- Zaman zaman trend değişimleri / volatilite patlamaları

Çalıştır:
  python simulate_backtest.py
  python simulate_backtest.py --candles 2000 --seed 99
"""

import argparse
import random
import math
import pandas as pd
from backtest import run_backtest, print_report
from config   import MA_FAST, MA_SLOW


def generate_price_series(n: int, seed: int = 42) -> pd.DataFrame:
    """Gerçekçi GBM tabanlı kripto fiyat serisi üret."""
    random.seed(seed)
    price   = 40_000.0    # BTC başlangıç fiyatı
    mu      = 0.0002      # saatlik drift (%0.02)
    sigma   = 0.012       # saatlik volatilite (%1.2)

    rows = []
    for i in range(n):
        # Periyodik volatilite patlaması (trend kırılması simülasyonu)
        if random.random() < 0.02:
            sigma = random.uniform(0.018, 0.030)
        else:
            sigma = max(0.008, sigma * 0.99)

        ret   = mu + sigma * _randn()
        price = price * math.exp(ret)

        high  = price * (1 + abs(_randn()) * sigma * 0.5)
        low   = price * (1 - abs(_randn()) * sigma * 0.5)
        open_ = price * (1 + _randn() * sigma * 0.2)
        vol   = random.uniform(100, 2000)

        rows.append({
            "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
            "open"  : open_,
            "high"  : max(high, open_, price),
            "low"   : min(low, open_, price),
            "close" : price,
            "volume": vol,
        })

    df = pd.DataFrame(rows)
    df.set_index("timestamp", inplace=True)
    return df


def _randn() -> float:
    """Box-Muller normal dağılım (stdlib ile)."""
    u1 = max(random.random(), 1e-10)
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candles", type=int, default=1500,
                        help="Simüle edilecek mum sayısı (default 1500 = ~62 gün)")
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--plot",    action="store_true")
    args = parser.parse_args()

    print(f"\nSimüle edilen {args.candles} saatlik mum verisi üretiliyor (seed={args.seed})…")
    df = generate_price_series(args.candles, seed=args.seed)

    print(f"Fiyat aralığı: {df['close'].min():,.0f} — {df['close'].max():,.0f} USDT\n")

    result = run_backtest(df, initial_balance=args.balance)
    print_report(result)

    # Detaylı işlem tablosu
    trades = result["trades"]
    if trades:
        print(f"\n{'#':>3}  {'Yön':<5}  {'Giriş':>10}  {'Çıkış':>10}  {'PnL':>10}  Neden")
        print("─" * 58)
        for i, t in enumerate(trades, 1):
            print(
                f"{i:>3}  {t['side'].upper():<5}  "
                f"{t['entry']:>10,.2f}  {t['exit']:>10,.2f}  "
                f"{t['pnl']:>+10.4f}  {t['reason']}"
            )

    if args.plot:
        from backtest import plot_equity
        plot_equity(result)


if __name__ == "__main__":
    main()
