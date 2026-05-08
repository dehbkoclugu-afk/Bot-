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

# --- MA periyotları ---
MA_FAST   = 21
MA_SLOW   = 55
MA_TREND  = 200
MA_TYPE   = "ema"
ATR_PERIOD = 14

# --- ADX filtresi ---
ADX_PERIOD    = 14
ADX_THRESHOLD = 20   # Gürültülü crossover'ları filtreler

# --- RSI (StochRSI için temel) ---
RSI_PERIOD     = 14
RSI_LONG_MIN   = 38
RSI_LONG_MAX   = 72
RSI_SHORT_MIN  = 28
RSI_SHORT_MAX  = 62

# --- Stochastic RSI giriş tetikleyicisi ---
SRSI_PERIOD    = 14    # RSI penceresi
SRSI_K_SMOOTH  = 3     # %K yumuşatma
SRSI_D_SMOOTH  = 3     # %D yumuşatma
SRSI_OVERSOLD  = 0.35  # Long: K bu seviyenin altından yukarı kesmeli
SRSI_OVERBOUGHT= 0.65  # Short: K bu seviyenin üstünden aşağı kesmeli

# --- Volume filtresi ---
VOLUME_MA_PERIOD = 20
VOLUME_MIN_MULT  = 1.2   # giriş mumundaki hacim ≥ ort. × 1.2

# --- Trailing ATR Stop ---
ATR_SL_MULT         = 1.2   # başlangıç SL: giriş ± 1.2×ATR
ATR_TRAIL_BREAKEVEN = 2.0   # +2.0×ATR'de SL → giriş fiyatına çek
ATR_TRAIL_ACTIVATE  = 3.0   # +3.0×ATR'de trailing başlar
ATR_TRAIL_DIST      = 1.5   # trailing: peak − 1.5×ATR
ATR_CAP_MULT        = 5.5   # emniyet TP kapağı (5.5×ATR)

# --- Çıkış modu ---
# EXIT_ON_CROSS=True → MA21/55 tersine döndüğünde pozisyonu kapat (trailing ile birlikte)
EXIT_ON_CROSS = True

# --- Giriş modu ---
# "ma_cross"   : EMA21/55 kesişiminde giriş, en sağlam mod
# "stochrsi"   : MA trend yönü + StochRSI aşırı satım geri dönüşü
# "rsi_pullback": MA trend yönü + RSI geri çekilme + toparlanma
ENTRY_MODE = "ma_cross"

# --- Risk ---
RISK_PER_TRADE  = 0.01
MAX_OPEN_TRADES = 1

# --- Bot döngüsü ---
POLL_INTERVAL_SEC = 60
CANDLES_REQUIRED  = 350

# --- Logging ---
LOG_FILE = "trading_bot.log"
