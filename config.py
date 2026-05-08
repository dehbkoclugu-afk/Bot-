import os
from dotenv import load_dotenv

load_dotenv()

# --- Exchange ---
API_KEY    = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
EXCHANGE   = os.getenv("EXCHANGE", "binance")
TESTNET    = os.getenv("TESTNET", "true").lower() == "true"

# --- Trading pair & timeframe ---
SYMBOL    = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = "1h"

# --- Strategy v2: EMA 21/55 + Regime + RSI + ADX ---
MA_FAST        = 21     # 21h hızlı EMA
MA_SLOW        = 55     # 55h yavaş EMA (Fibonacci)
MA_TREND       = 200    # 200h rejim filtresi
MA_TYPE        = "ema"
ATR_PERIOD     = 14

# Filtreler
ADX_PERIOD     = 14
ADX_THRESHOLD  = 20     # yalnızca trendin gücü bu seviyenin üzerindeyken işlem aç
RSI_PERIOD     = 14
RSI_LONG_MIN   = 38     # long sinyali için RSI alt sınırı
RSI_LONG_MAX   = 72     # long sinyali için RSI üst sınırı (aşırı alım)
RSI_SHORT_MIN  = 28     # short sinyali için ters RSI sınırı
RSI_SHORT_MAX  = 62

# Çıkış filtresi
EXIT_ON_CROSS  = True   # MA çaprazı ters dönerse pozisyonu kapat

# --- Risk management ---
RISK_PER_TRADE   = 0.01   # bakiyenin %1'i
ATR_SL_MULT      = 1.5    # stop-loss = 1.5 × ATR
ATR_TP_MULT      = 3.5    # take-profit = 3.5 × ATR  (2.3:1 RR)
MAX_OPEN_TRADES  = 1

# --- Bot loop ---
POLL_INTERVAL_SEC = 60
CANDLES_REQUIRED  = 300   # 200 EMA için yeterli geçmiş

# --- Logging ---
LOG_FILE = "trading_bot.log"
