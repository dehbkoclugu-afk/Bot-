"""
Crypto Trading Bot — MA Crossover (5h / 12h)
=============================================

Strategy
--------
  BUY  when 5-hour MA crosses ABOVE 12-hour MA (golden cross)
  SELL when 5-hour MA crosses BELOW 12-hour MA (death  cross)

Risk Management
---------------
  Stop-Loss   : entry − 1.5 × ATR   (long)
  Take-Profit : entry + 3.0 × ATR   (long)  → 2:1 reward-to-risk
  Position    : 1% of account equity risked per trade

Run
---
  python bot.py              # live trading (configure .env first)
  python bot.py --dry-run    # paper trading (no real orders)
"""

import argparse
import logging
import time
import json
from datetime import datetime
from pathlib import Path

import exchange_client as ex
from strategy     import get_signal
from risk_manager import calculate_levels, calculate_position_size, check_exit
from config       import (
    SYMBOL, TIMEFRAME, MA_FAST, MA_SLOW, MA_TYPE,
    ATR_SL_MULT, ATR_TP_MULT, RISK_PER_TRADE,
    POLL_INTERVAL_SEC, MAX_OPEN_TRADES, LOG_FILE,
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
# State file (survives restarts)
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
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Core trading logic
# ---------------------------------------------------------------------------

class TradingBot:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.state   = _load_state()
        mode = "DRY-RUN" if dry_run else "LIVE"
        log.info("=" * 60)
        log.info("Bot started  [%s]  %s  %s/%sh  MA%d×MA%d",
                 mode, SYMBOL, MA_FAST, MA_SLOW, MA_FAST, MA_SLOW)
        log.info("SL=%.1f×ATR  TP=%.1f×ATR  Risk=%.0f%%/trade",
                 ATR_SL_MULT, ATR_TP_MULT, RISK_PER_TRADE * 100)
        log.info("=" * 60)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> dict | None:
        return self.state.get("position")

    @position.setter
    def position(self, val):
        self.state["position"] = val
        _save_state(self.state)

    # ------------------------------------------------------------------
    # Order helpers (dry-run aware)
    # ------------------------------------------------------------------

    def _open_long(self, price: float, qty: float, tp: float, sl: float):
        log.info(">>> OPEN LONG  qty=%.6f  entry=%.4f  TP=%.4f  SL=%.4f",
                 qty, price, tp, sl)
        if not self.dry_run:
            order = ex.place_market_order("buy", qty)
            if not order:
                return
            price = float(order.get("average") or price)

        self.position = {
            "side"        : "buy",
            "entry"       : price,
            "qty"         : qty,
            "take_profit" : tp,
            "stop_loss"   : sl,
            "opened_at"   : datetime.utcnow().isoformat(),
        }

    def _open_short(self, price: float, qty: float, tp: float, sl: float):
        log.info(">>> OPEN SHORT qty=%.6f  entry=%.4f  TP=%.4f  SL=%.4f",
                 qty, price, tp, sl)
        if not self.dry_run:
            order = ex.place_market_order("sell", qty)
            if not order:
                return
            price = float(order.get("average") or price)

        self.position = {
            "side"        : "sell",
            "entry"       : price,
            "qty"         : qty,
            "take_profit" : tp,
            "stop_loss"   : sl,
            "opened_at"   : datetime.utcnow().isoformat(),
        }

    def _close_position(self, current_price: float, reason: str):
        pos = self.position
        if pos is None:
            return

        side     = pos["side"]
        entry    = pos["entry"]
        qty      = pos["qty"]
        close_side = "sell" if side == "buy" else "buy"

        pnl_pct = ((current_price - entry) / entry * 100) * (1 if side == "buy" else -1)
        pnl_abs = (current_price - entry) * qty * (1 if side == "buy" else -1)

        log.info("<<< CLOSE %s  reason=%s  exit=%.4f  PnL=%.2f%%  (%.4f USDT)",
                 side.upper(), reason, current_price, pnl_pct, pnl_abs)

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
    # Stats
    # ------------------------------------------------------------------

    def _print_stats(self):
        trades = self.state.get("trades", [])
        if not trades:
            return
        wins  = [t for t in trades if t["pnl_abs"] > 0]
        losses = [t for t in trades if t["pnl_abs"] <= 0]
        total_pnl = sum(t["pnl_abs"] for t in trades)
        win_rate  = len(wins) / len(trades) * 100 if trades else 0
        log.info("--- Stats: %d trades | WR=%.0f%% | Total PnL=%.4f USDT | W=%d L=%d",
                 len(trades), win_rate, total_pnl, len(wins), len(losses))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        log.info("Bot running. Press Ctrl+C to stop.")
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                log.info("Bot stopped by user.")
                self._print_stats()
                break
            except Exception as exc:
                log.exception("Unexpected error: %s — retrying next tick", exc)
            time.sleep(POLL_INTERVAL_SEC)

    def _tick(self):
        # 1. Fetch market data
        df = ex.fetch_ohlcv()
        if df is None or df.empty:
            log.warning("No OHLCV data received — skipping tick")
            return

        current_price = ex.fetch_ticker_price()
        if current_price is None:
            log.warning("Could not fetch current price — skipping tick")
            return

        # 2. Evaluate open position for TP/SL
        if self.position:
            pos    = self.position
            result = check_exit(
                pos["side"], pos["entry"],
                current_price, pos["take_profit"], pos["stop_loss"],
            )
            if result != "hold":
                self._close_position(current_price, result)
                return

            log.debug(
                "Position OPEN [%s]  entry=%.4f  curr=%.4f  TP=%.4f  SL=%.4f",
                pos["side"], pos["entry"], current_price,
                pos["take_profit"], pos["stop_loss"],
            )
            return   # don't open another position while one is active

        # 3. Check for signal (no open position)
        if len(self.state.get("trades", [])) >= MAX_OPEN_TRADES * 10:
            pass   # no hard limit on total trades

        sig = get_signal(df)
        log.info(
            "Candle check  price=%.4f  MA%d=%.4f  MA%d=%.4f  signal=%s",
            current_price,
            MA_FAST, sig.get("ma_fast", 0),
            MA_SLOW, sig.get("ma_slow", 0),
            sig["signal"].upper(),
        )

        if sig["signal"] == "none":
            return

        # 4. Calculate TP / SL / position size
        side   = sig["signal"]     # 'buy' or 'sell'
        price  = sig["price"]
        atr    = sig["atr"]
        levels = calculate_levels(price, atr, side)
        tp     = levels["take_profit"]
        sl     = levels["stop_loss"]

        balance = ex.fetch_balance() if not self.dry_run else 10_000.0
        qty     = calculate_position_size(balance, price, sl)

        if qty <= 0:
            log.warning("Calculated qty is zero — skipping signal")
            return

        log.info(
            "Signal: %s  price=%.4f  TP=%.4f  SL=%.4f  RR=%.2f  qty=%.6f",
            side.upper(), price, tp, sl, levels["rr_ratio"], qty,
        )

        # 5. Open position
        if side == "buy":
            self._open_long(price, qty, tp, sl)
        else:
            self._open_short(price, qty, tp, sl)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MA Crossover Crypto Trading Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Paper trading — signals are logged but no real orders are placed",
    )
    args = parser.parse_args()

    bot = TradingBot(dry_run=args.dry_run)
    bot.run()


if __name__ == "__main__":
    main()
