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
MA_FAST    = 21   # orta: EMA21 (macd_score_rider için)
MA_SLOW    = 55   # yavaş: EMA55
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

# --- MACD göstergesi (macd_score_rider için) ---
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

# --- Market Skor sistemi (macd_score_rider) ---
SCORE_MIN  = 6    # minimum giriş skoru (0-10)
SCORE_HIGH = 9    # bu skorun üstünde RISK_HIGH kullanılır

# --- Supertrend parametreleri ---
ST_ATR_PERIOD = 14   # ATR periyodu
ST_MULTIPLIER = 7.0  # Bant genişliği (yüksek = daha az flip, trende daha uzun kalır)

# --- Volume filtresi ---
VOLUME_MA_PERIOD = 20
VOLUME_MIN_MULT  = 1.0   # macd_score_rider: 1.0× (daha gevşek); diğerleri: 1.2×

# --- Trailing ATR Stop ---
ATR_SL_MULT         = 2.0    # başlangıç SL: giriş ± 2.0×ATR
ATR_TRAIL_BREAKEVEN = 2.5    # +2.5×ATR'de SL → giriş fiyatına çek
ATR_TRAIL_ACTIVATE  = 4.5    # +4.5×ATR'de trailing başlar
ATR_TRAIL_DIST      = 3.0    # trailing: peak − 3.0×ATR
ATR_CAP_MULT        = 20.0   # emniyet TP kapağı (neredeyse hiç tetiklenmez)

# --- Çıkış modu ---
EXIT_ON_CROSS = False   # macd_score_rider: yalnızca trailing stop ile çıkış

# --- Giriş modu ---
# "supertrend"      : Supertrend flip (mult=7×ATR14) + EMA200 rejim — en kaliteli (KazO=65%, PF=3.58)
# "quad_ma"         : EMA5/15/50/200 tam hizalanınca giriş — 5 seed ort. +2.42%
# "ma_cross"        : EMA21/55 kesişiminde giriş
# "macd_score_rider": MACD histogram + piyasa skoru (yüksek sinyal sayısı, düşük kalite)
# "stochrsi"        : MA trend yönü + StochRSI aşırı satım geri dönüşü
# "rsi_pullback"    : MA trend yönü + RSI geri çekilme + toparlanma
ENTRY_MODE = "supertrend"

# --- Risk ---
RISK_PER_TRADE  = 0.02   # %2 — Supertrend'in yüksek kalitesinden faydalanmak için artırıldı
RISK_HIGH       = 0.05   # macd_score_rider: skor ≥ SCORE_HIGH girişleri
RISK_LOW        = 0.01   # macd_score_rider: skor 6-8 girişleri
MAX_OPEN_TRADES = 1

# --- Bot döngüsü ---
POLL_INTERVAL_SEC = 60
CANDLES_REQUIRED  = 350

# --- Logging ---
LOG_FILE = "trading_bot.log"
