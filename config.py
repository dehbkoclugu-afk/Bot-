import os
from dotenv import load_dotenv

load_dotenv()

# --- Exchange ---
API_KEY    = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
EXCHANGE   = os.getenv("EXCHANGE", "binance")   # ccxt exchange id
TESTNET    = os.getenv("TESTNET", "true").lower() == "true"

# --- Trading pair & timeframe ---
SYMBOL    = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = "1h"          # 1-hour candles → 5 bar MA=5h, 12 bar MA=12h

# --- Strategy ---
MA_FAST   = 5             # 5-hour fast MA period
MA_SLOW   = 12            # 12-hour slow MA period
MA_TYPE   = "ema"         # "ema" or "sma"
ATR_PERIOD = 14           # ATR period for dynamic TP/SL

# --- Risk management ---
RISK_PER_TRADE   = 0.01   # 1% of balance risked per trade
ATR_SL_MULT      = 1.5    # Stop-loss = 1.5 × ATR below entry
ATR_TP_MULT      = 3.0    # Take-profit = 3.0 × ATR above entry (2:1 RR)
MAX_OPEN_TRADES  = 1      # Only one position at a time

# --- Bot loop ---
POLL_INTERVAL_SEC = 60    # Check signals every 60 seconds
CANDLES_REQUIRED  = 50    # History candles to fetch (> MA_SLOW)

# --- Logging ---
LOG_FILE = "trading_bot.log"
