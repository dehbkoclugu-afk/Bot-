"""
Kripto Trading Botu — EMA 21/55 + 200 Rejim Filtresi
======================================================

Strateji Özeti
--------------
  LONG  : EMA21 EMA55'i yukari keser + fiyat EMA200 üstünde + RSI[38-72] + ADX≥20
  SHORT : EMA21 EMA55'i asagi keser + fiyat EMA200 altinda + RSI[28-62] + ADX≥20

Risk Yönetimi
-------------
  Stop-Loss   : giriş − 1.5 × ATR
  Take-Profit : giriş + 3.5 × ATR  (2.3:1 ödül/risk)
  Pozisyon    : bakiyenin %1'i her işlemde riske edilir
  Erken Çıkış: MA tersine kesişirse pozisyon kapatılır

Çalıştırma
----------
  python bot.py              # canlı
  python bot.py --dry-run    # kağıt üzerinde (gerçek emir yok)
  python bot.py --dry-run --long-only   # yalnızca long (spot cüzdan için)
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import exchange_client as ex
from strategy     import get_signal, should_exit_early
from risk_manager import calculate_levels, calculate_position_size, check_exit
from config       import (
    SYMBOL, TIMEFRAME, MA_FAST, MA_SLOW, MA_TREND,
    ATR_SL_MULT, ATR_TP_MULT, RISK_PER_TRADE,
    POLL_INTERVAL_SEC, CANDLES_REQUIRED, EXIT_ON_CROSS, LOG_FILE,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kalıcı durum
# ---------------------------------------------------------------------------

STATE_FILE = Path("state.json")


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"position": None, "trades": []}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class TradingBot:
    def __init__(self, dry_run: bool = False, long_only: bool = False):
        self.dry_run   = dry_run
        self.long_only = long_only
        self.state     = _load_state()
        mode = "DRY-RUN" if dry_run else "LIVE"
        lo   = "  [LONG-ONLY]" if long_only else ""
        log.info("=" * 64)
        log.info("Bot başlatıldı  [%s]%s  %s", mode, lo, SYMBOL)
        log.info("Strateji: EMA%d/EMA%d  Rejim: EMA%d", MA_FAST, MA_SLOW, MA_TREND)
        log.info("SL=%.1f×ATR  TP=%.1f×ATR  Risk=%.0f%%/işlem",
                 ATR_SL_MULT, ATR_TP_MULT, RISK_PER_TRADE * 100)
        log.info("=" * 64)

    @property
    def position(self) -> dict | None:
        return self.state.get("position")

    @position.setter
    def position(self, val):
        self.state["position"] = val
        _save_state(self.state)

    # ------------------------------------------------------------------
    # Emir yardımcıları
    # ------------------------------------------------------------------

    def _open_position(self, side: str, price: float, qty: float,
                       tp: float, sl: float):
        log.info(">>> PozisYon AÇ [%s]  qty=%.6f  giriş=%.2f  TP=%.2f  SL=%.2f",
                 side.upper(), qty, price, tp, sl)

        if not self.dry_run:
            order = ex.place_market_order(side, qty)
            if not order:
                log.error("Emir verilemedi — pozisyon açılmadı")
                return
            price = float(order.get("average") or price)

        self.position = {
            "side"      : side,
            "entry"     : price,
            "qty"       : qty,
            "take_profit": tp,
            "stop_loss" : sl,
            "opened_at" : datetime.utcnow().isoformat(),
        }

    def _close_position(self, current_price: float, reason: str):
        pos = self.position
        if pos is None:
            return

        side       = pos["side"]
        entry      = pos["entry"]
        qty        = pos["qty"]
        close_side = "sell" if side == "buy" else "buy"
        direction  = 1 if side == "buy" else -1

        pnl_pct = (current_price - entry) / entry * 100 * direction
        pnl_abs = (current_price - entry) * qty * direction

        log.info(
            "<<< PozisYon KAP [%s]  neden=%-12s  çıkış=%.2f  PnL=%+.2f%%  (%+.4f USDT)",
            side.upper(), reason, current_price, pnl_pct, pnl_abs,
        )

        if not self.dry_run:
            ex.place_market_order(close_side, qty)

        self.state["trades"].append({
            "side"      : side,
            "entry"     : entry,
            "exit"      : current_price,
            "qty"       : qty,
            "pnl_pct"   : round(pnl_pct, 4),
            "pnl_abs"   : round(pnl_abs, 4),
            "reason"    : reason,
            "closed_at" : datetime.utcnow().isoformat(),
        })
        self.position = None
        _save_state(self.state)
        self._print_stats()

    # ------------------------------------------------------------------
    # İstatistik
    # ------------------------------------------------------------------

    def _print_stats(self):
        trades = self.state.get("trades", [])
        if not trades:
            return
        wins      = [t for t in trades if t["pnl_abs"] > 0]
        losses    = [t for t in trades if t["pnl_abs"] <= 0]
        total_pnl = sum(t["pnl_abs"] for t in trades)
        win_rate  = len(wins) / len(trades) * 100
        profit_factor = (
            abs(sum(t["pnl_abs"] for t in wins) / sum(t["pnl_abs"] for t in losses))
            if losses and sum(t["pnl_abs"] for t in losses) != 0 else float("inf")
        )
        log.info(
            "─── İstatistik: %d işlem | KazO=%.0f%% | PF=%.2f | Toplam PnL=%+.4f USDT",
            len(trades), win_rate, profit_factor, total_pnl,
        )

    # ------------------------------------------------------------------
    # Ana döngü
    # ------------------------------------------------------------------

    def run(self):
        log.info("Bot çalışıyor. Durdurmak için Ctrl+C basın.")
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                log.info("Bot kullanıcı tarafından durduruldu.")
                self._print_stats()
                break
            except Exception as exc:
                log.exception("Beklenmedik hata: %s — sonraki turda tekrar denenecek", exc)
            time.sleep(POLL_INTERVAL_SEC)

    def _tick(self):
        df = ex.fetch_ohlcv(limit=CANDLES_REQUIRED)
        if df is None or df.empty:
            log.warning("OHLCV verisi alınamadı — tur atlanıyor")
            return

        current_price = ex.fetch_ticker_price()
        if current_price is None:
            log.warning("Güncel fiyat alınamadı — tur atlanıyor")
            return

        # 1. Açık pozisyon kontrolü
        if self.position:
            pos    = self.position
            result = check_exit(
                pos["side"], pos["entry"],
                current_price, pos["take_profit"], pos["stop_loss"],
            )
            if result != "hold":
                self._close_position(current_price, result)
                return

            # Erken çıkış: MA tersine döndüyse
            if EXIT_ON_CROSS and should_exit_early(df, pos["side"]):
                self._close_position(current_price, "ma_cross_exit")
                return

            log.info(
                "Açık [%s]  giriş=%.2f  güncel=%.2f  TP=%.2f  SL=%.2f",
                pos["side"].upper(), pos["entry"], current_price,
                pos["take_profit"], pos["stop_loss"],
            )
            return

        # 2. Yeni sinyal kontrolü
        sig = get_signal(df)

        bull_str = "boğa" if sig.get("bull") else "ayı"
        log.info(
            "Analiz  fiyat=%.2f  EMA%d=%.2f  EMA%d=%.2f  RSI=%.1f  ADX=%.1f  rejim=%s  sinyal=%s",
            current_price,
            MA_FAST, sig.get("ma_fast", 0),
            MA_SLOW, sig.get("ma_slow", 0),
            sig.get("rsi", 0), sig.get("adx", 0),
            bull_str, sig["signal"].upper(),
        )

        if sig["signal"] == "none":
            return

        # Long-only modunda short sinyali yok say
        if self.long_only and sig["signal"] == "sell":
            log.info("Long-only mod: SELL sinyali yok sayıldı")
            return

        side   = sig["signal"]
        price  = sig["price"]
        atr    = sig["atr"]
        levels = calculate_levels(price, atr, side)
        tp     = levels["take_profit"]
        sl     = levels["stop_loss"]

        balance = 10_000.0 if self.dry_run else ex.fetch_balance()
        qty     = calculate_position_size(balance, price, sl)

        if qty <= 0:
            log.warning("Pozisyon boyutu sıfır — sinyal atlanıyor")
            return

        log.info(
            "SİNYAL %s  fiyat=%.2f  TP=%.2f  SL=%.2f  RR=%.2f  qty=%.6f",
            side.upper(), price, tp, sl, levels["rr_ratio"], qty,
        )

        self._open_position(side, price, qty, tp, sl)


# ---------------------------------------------------------------------------
# Başlangıç noktası
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EMA 21/55 Kripto Trading Botu")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Kağıt üzerinde test — gerçek emir verilmez",
    )
    parser.add_argument(
        "--long-only", action="store_true",
        help="Yalnızca LONG pozisyon aç (spot cüzdan için)",
    )
    args = parser.parse_args()

    bot = TradingBot(dry_run=args.dry_run, long_only=args.long_only)
    bot.run()


if __name__ == "__main__":
    main()
