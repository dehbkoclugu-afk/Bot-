"""
Kripto Trading Botu v3 — Trailing ATR Stop + StochRSI Giriş
=============================================================

Strateji
--------
  LONG : EMA21 > EMA55 + boğa rejimi (EMA200) + ADX≥25 + StochRSI oversold çıkışı + Volume
  SHORT: EMA21 < EMA55 + ayı rejimi + ADX≥25 + StochRSI overbought çıkışı + Volume

Çıkış
-----
  Trailing ATR Stop: +1.5×ATR breakeven, +2.5×ATR trailing @ peak − 1.5×ATR
  Emniyet kapağı: 8×ATR (pratik olarak ulaşılmaz, yalnızca güvenlik için)

Çalıştırma
----------
  python bot.py --dry-run              # kağıt üzerinde test
  python bot.py --dry-run --long-only  # sadece long (spot cüzdan)
  python bot.py                        # canlı (.env ayarlanmış olmalı)
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import exchange_client as ex
from strategy     import get_signal
from risk_manager import (
    calculate_levels, calculate_position_size,
    update_trailing_stop, check_exit,
)
from config import (
    SYMBOL, TIMEFRAME, MA_FAST, MA_SLOW, MA_TREND, ENTRY_MODE, EXIT_ON_CROSS,
    ATR_SL_MULT, ATR_CAP_MULT, ATR_TRAIL_BREAKEVEN, ATR_TRAIL_ACTIVATE, ATR_TRAIL_DIST,
    RISK_PER_TRADE, POLL_INTERVAL_SEC, CANDLES_REQUIRED, LOG_FILE,
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
    return {"position": None, "trades": [], "pb_state": {}}


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

        # StochRSI durum makinesi kalıcı tutulur
        if "pb_state" not in self.state:
            self.state["pb_state"] = {}
        self.pb_state = self.state["pb_state"]
        if not self.pb_state:
            self.pb_state.update({
                "long_armed": False, "long_seen": False,
                "short_armed": False, "short_seen": False,
            })

        mode = "DRY-RUN" if dry_run else "LIVE"
        lo   = "  [LONG-ONLY]" if long_only else ""
        log.info("=" * 64)
        log.info("Bot v3 başlatıldı  [%s]%s  %s", mode, lo, SYMBOL)
        log.info("Strateji: EMA%d/EMA%d  Rejim:EMA%d  Giriş:%s",
                 MA_FAST, MA_SLOW, MA_TREND, ENTRY_MODE)
        log.info("SL=%.1f×ATR  CapTP=%.1f×ATR  Trail:BE=%.1f/Act=%.1f/Dist=%.1f",
                 ATR_SL_MULT, ATR_CAP_MULT,
                 ATR_TRAIL_BREAKEVEN, ATR_TRAIL_ACTIVATE, ATR_TRAIL_DIST)
        log.info("Risk=%.0f%%/işlem", RISK_PER_TRADE * 100)
        log.info("=" * 64)

        # Eski state formatını yeni formata geçir
        self._migrate_state()

    def _migrate_state(self):
        """Eski state.json formatından yeni formata geçiş."""
        pos = self.state.get("position")
        if pos and "take_profit" in pos:
            atr_guess = pos["entry"] * 0.01
            pos["cap_tp"]   = pos.pop("take_profit")
            pos["stop_loss_key"] = pos.pop("stop_loss", pos["entry"] - 2 * atr_guess)
            pos["atr_e"]    = atr_guess
            pos["extreme"]  = pos["entry"]
            self.state["position"] = pos
            _save_state(self.state)

    # ------------------------------------------------------------------
    # Özellikler
    # ------------------------------------------------------------------

    @property
    def position(self) -> dict | None:
        return self.state.get("position")

    @position.setter
    def position(self, val):
        self.state["position"] = val
        self.state["pb_state"] = self.pb_state
        _save_state(self.state)

    # ------------------------------------------------------------------
    # Emir yardımcıları
    # ------------------------------------------------------------------

    def _open_position(self, side: str, price: float, qty: float,
                       sl: float, cap_tp: float, atr: float):
        log.info(
            ">>> PozisYon AÇ [%s]  qty=%.6f  giriş=%.2f  SL=%.2f  CapTP=%.2f",
            side.upper(), qty, price, sl, cap_tp,
        )
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
            "sl"        : sl,
            "cap_tp"    : cap_tp,
            "atr_e"     : atr,
            "extreme"   : price,
            "opened_at" : datetime.utcnow().isoformat(),
        }

    def _close_position(self, current_price: float, reason: str):
        pos = self.position
        if pos is None:
            return

        side      = pos["side"]
        entry     = pos["entry"]
        qty       = pos["qty"]
        close_sd  = "sell" if side == "buy" else "buy"
        direction = 1 if side == "buy" else -1

        pnl_pct = (current_price - entry) / entry * 100 * direction
        pnl_abs = (current_price - entry) * qty * direction
        peak_move = abs(pos["extreme"] - entry)

        log.info(
            "<<< PozisYon KAP [%s]  neden=%-10s  çıkış=%.2f  PnL=%+.2f%%  (%+.2f USDT)  peak_atr=%.1f×",
            side.upper(), reason, current_price, pnl_pct, pnl_abs,
            peak_move / pos["atr_e"] if pos["atr_e"] else 0,
        )

        if not self.dry_run:
            ex.place_market_order(close_sd, qty)

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
        pf        = (
            abs(sum(t["pnl_abs"] for t in wins) / sum(t["pnl_abs"] for t in losses))
            if losses and sum(t["pnl_abs"] for t in losses) != 0 else float("inf")
        )
        log.info(
            "─── İstatistik: %d işlem | KazO=%.0f%% | PF=%.2f | PnL=%+.2f USDT",
            len(trades), win_rate, pf, total_pnl,
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
                log.info("Bot durduruldu.")
                self._print_stats()
                break
            except Exception as exc:
                log.exception("Beklenmedik hata: %s", exc)
            time.sleep(POLL_INTERVAL_SEC)

    def _tick(self):
        df = ex.fetch_ohlcv(limit=CANDLES_REQUIRED)
        if df is None or df.empty:
            log.warning("OHLCV verisi alınamadı")
            return

        current_price = ex.fetch_ticker_price()
        if current_price is None:
            log.warning("Güncel fiyat alınamadı")
            return

        # Anlık ATR (trailing stop için)
        from strategy import add_indicators
        df_ind = add_indicators(df)
        current_atr = float(df_ind["atr"].iloc[-1])

        # ----------------------------------------------------------------
        # Açık pozisyon: trailing stop güncelle + çıkış kontrolü
        # ----------------------------------------------------------------
        if self.position:
            pos = self.position

            new_sl, new_ext = update_trailing_stop(
                pos["side"], pos["entry"], pos["atr_e"],
                current_price, pos["sl"], pos["extreme"],
            )
            pos["sl"]      = new_sl
            pos["extreme"] = new_ext
            self.position  = pos

            result = check_exit(
                pos["side"], current_price, pos["sl"], pos["cap_tp"]
            )
            if result != "hold":
                self._close_position(current_price, result)
                return

            # EXIT_ON_CROSS: Son kapanan mumda MA tersine döndüyse kapat
            if EXIT_ON_CROSS:
                ma_fast_now = float(df_ind["ma_fast"].iloc[-2])  # son kapanan
                ma_slow_now = float(df_ind["ma_slow"].iloc[-2])
                ma_fast_prv = float(df_ind["ma_fast"].iloc[-3])  # önceki kapanan
                ma_slow_prv = float(df_ind["ma_slow"].iloc[-3])
                cross_down = ma_fast_prv > ma_slow_prv and ma_fast_now <= ma_slow_now
                cross_up   = ma_fast_prv < ma_slow_prv and ma_fast_now >= ma_slow_now
                if (pos["side"] == "buy" and cross_down) or (pos["side"] == "sell" and cross_up):
                    self._close_position(current_price, "cross_exit")
                    return

            log.info(
                "Açık [%s]  giriş=%.2f  güncel=%.2f  SL=%.2f  Peak=%.2f  ATR×=%.1f",
                pos["side"].upper(), pos["entry"], current_price,
                pos["sl"], pos["extreme"],
                abs(pos["extreme"] - pos["entry"]) / pos["atr_e"] if pos["atr_e"] else 0,
            )
            return

        # ----------------------------------------------------------------
        # Sinyal kontrolü (StochRSI durum makinesi ile)
        # ----------------------------------------------------------------
        sig = get_signal(df, state=self.pb_state)

        log.info(
            "Analiz  %.2f  EMA%d=%.2f  ADX=%.1f  K=%.2f/D=%.2f  rejim=%s  sinyal=%s",
            current_price,
            MA_FAST, sig.get("ma_fast", 0),
            sig.get("adx", 0),
            sig.get("srsi_k", 0), sig.get("srsi_d", 0),
            "boğa" if sig.get("bull") else "ayı",
            sig["signal"].upper(),
        )

        if sig["signal"] == "none":
            return

        if self.long_only and sig["signal"] == "sell":
            log.info("Long-only mod: SELL sinyali yok sayıldı")
            return

        side   = sig["signal"]
        price  = sig["price"]
        atr    = sig["atr"]
        levels = calculate_levels(price, atr, side)
        sl     = levels["stop_loss"]
        cap_tp = levels["cap_tp"]

        balance = 10_000.0 if self.dry_run else ex.fetch_balance()
        qty     = calculate_position_size(balance, price, sl)

        if qty <= 0:
            log.warning("Pozisyon boyutu sıfır — sinyal atlanıyor")
            return

        log.info(
            "SİNYAL %s  fiyat=%.2f  SL=%.2f  CapTP=%.2f  qty=%.6f",
            side.upper(), price, sl, cap_tp, qty,
        )

        self._open_position(side, price, qty, sl, cap_tp, atr)


# ---------------------------------------------------------------------------
# Başlangıç
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trading Bot v3")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--long-only", action="store_true")
    args = parser.parse_args()
    TradingBot(dry_run=args.dry_run, long_only=args.long_only).run()


if __name__ == "__main__":
    main()
