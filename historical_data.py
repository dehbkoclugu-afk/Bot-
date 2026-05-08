"""
BTC/USDT 2024 Gerçekçi Tarihsel Veri Üretici
----------------------------------------------
2024 yılı boyunca BTC'nin gerçek fiyat hareketlerini
(40k→73k ATH→boğa sonu→seçim pompası→108k) birebir taklit eder.

Gerçek pivot noktaları referans alındı:
  Jan  2024: ~42,000
  Feb  2024: ~62,000 (büyük ralli)
  Mar  2024: ~73,750 (ATH)
  Apr  2024: ~60,000 (düzeltme)
  May  2024: ~67,000
  Jun  2024: ~57,000 (geri çekilme)
  Jul  2024: ~66,000
  Aug  2024: ~58,000
  Sep  2024: ~63,500
  Oct  2024: ~72,000
  Nov  2024: ~99,000 (seçim pompası)
  Dec  2024: ~106,000
"""

import math
import random
import pandas as pd


# Her ay için (başlangıç_fiyat, bitiş_fiyat, volatilite_çarpanı) tuple'ları
MONTHLY_PIVOTS = [
    # (ay_başı, ay_sonu, vol_multiplier)
    (42_000,  45_500, 1.0),   # Ocak
    (45_500,  62_000, 1.4),   # Şubat — güçlü ralli
    (62_000,  73_750, 1.6),   # Mart — ATH tırmanışı
    (73_750,  60_000, 1.8),   # Nisan — sert düzeltme
    (60_000,  67_000, 1.1),   # Mayıs — toparlanma
    (67_000,  57_000, 1.3),   # Haziran — geri çekilme
    (57_000,  66_000, 1.2),   # Temmuz — sıçrama
    (66_000,  58_000, 1.2),   # Ağustos — düzeltme
    (58_000,  63_500, 0.9),   # Eylül — yatay
    (63_500,  72_000, 1.1),   # Ekim — yükseliş
    (72_000,  99_000, 2.0),   # Kasım — seçim pompası
    (99_000, 106_000, 1.5),   # Aralık — devam eden boğa
]

HOURS_PER_MONTH = 730   # yaklaşık


def _randn(rng: random.Random) -> float:
    u1 = max(rng.random(), 1e-10)
    u2 = rng.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def generate_btc_2024(seed: int = 0) -> pd.DataFrame:
    """
    2024 BTC/USDT saatlik mum verisi üret (≈8 760 satır).

    Her ay sonunda fiyat, hedef değere yakınsatılır (Brownian bridge).
    Bu sayede gerçek 2024 BTC fiyat aralığı (38k–108k) korunur.
    """
    rng = random.Random(seed)
    rows = []
    timestamp = pd.Timestamp("2024-01-01")

    for price_start, price_end, vol_mul in MONTHLY_PIVOTS:
        hours = HOURS_PER_MONTH
        base_sigma = 0.009 * vol_mul  # saatlik volatilite

        log_start = math.log(price_start)
        log_end   = math.log(price_end)

        # Brownian bridge: her adımda kalan mesafeye doğru drift ekle
        log_price = log_start

        for h in range(hours):
            remaining = hours - h
            # Hedefe doğru düzeltme drift'i
            drift = (log_end - log_price) / remaining

            # Volatilite şoku (düşük olasılıklı)
            sigma = base_sigma * (rng.uniform(2, 3) if rng.random() < 0.012 else 1.0)

            log_price += drift + sigma * _randn(rng)
            price = math.exp(log_price)

            wick  = abs(_randn(rng)) * sigma * price * 0.5
            open_ = price * (1 + _randn(rng) * sigma * 0.25)
            high  = max(price, open_) + wick
            low   = min(price, open_) - max(wick * 0.8, 0)
            vol   = rng.uniform(200, 3_000) * vol_mul

            rows.append({
                "timestamp": timestamp,
                "open"  : max(open_, 1),
                "high"  : max(high,  1),
                "low"   : max(low,   1),
                "close" : max(price, 1),
                "volume": vol,
            })
            timestamp += pd.Timedelta(hours=1)

    df = pd.DataFrame(rows)
    df.set_index("timestamp", inplace=True)
    return df


if __name__ == "__main__":
    df = generate_btc_2024()
    print(f"Toplam mum: {len(df)}")
    print(f"Dönem     : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Min fiyat : {df['close'].min():,.0f} USD")
    print(f"Max fiyat : {df['close'].max():,.0f} USD")
    print(df[["open","high","low","close","volume"]].tail(5).to_string())
