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

# --- Dörtlü MA (quad_ma) periyotları: EMA5 > EMA15 > EMA50 > EMA200 ---
MA_FASTEST = 5    # tetikleyici: EMA5
MA_FAST    = 15   # onay 1: EMA15
MA_SLOW    = 50   # onay 2: EMA50
MA_TREND   = 200  # rejim: EMA200
MA_TYPE    = "ema"
ATR_PERIOD = 14

# --- ADX filtresi ---
ADX_PERIOD    = 14
ADX_THRESHOLD = 20   # trend gücü eşiği

# --- RSI (StochRSI için temel) ---
RSI_PERIOD     = 14
RSI_LONG_MIN   = 38
RSI_LONG_MAX   = 72
RSI_SHORT_MIN  = 28
RSI_SHORT_MAX  = 62

# --- Stochastic RSI giriş tetikleyicisi ---
SRSI_PERIOD    = 14
SRSI_K_SMOOTH  = 3
SRSI_D_SMOOTH  = 3
SRSI_OVERSOLD  = 0.35
SRSI_OVERBOUGHT= 0.65

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
# EXIT_ON_CROSS=True → hızlı MA / ikinci MA tersine döndüğünde pozisyonu kapat
EXIT_ON_CROSS = True

# --- Giriş modu ---
# "quad_ma"    : EMA5/15/50/200 tam hizalanınca giriş — 5 seed'de en tutarlı (+2.4% ort.)
# "ma_cross"   : EMA15/50 kesişiminde giriş — en yüksek peak getiri (+19% seed0)
# "stochrsi"   : MA trend yönü + StochRSI aşırı satım geri dönüşü
# "rsi_pullback": MA trend yönü + RSI geri çekilme + toparlanma
ENTRY_MODE = "quad_ma"

# --- Risk ---
RISK_PER_TRADE  = 0.01
MAX_OPEN_TRADES = 1

# --- Bot döngüsü ---
POLL_INTERVAL_SEC = 60
CANDLES_REQUIRED  = 350

# --- Logging ---
LOG_FILE = "trading_bot.log"
