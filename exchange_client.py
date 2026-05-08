"""
Exchange Client — thin wrapper around ccxt.
Handles OHLCV fetching and order placement with unified error handling.
"""

import logging
import ccxt
import pandas as pd
from config import (
    API_KEY, API_SECRET, EXCHANGE, TESTNET,
    SYMBOL, TIMEFRAME, CANDLES_REQUIRED,
)

log = logging.getLogger(__name__)


def _build_exchange() -> ccxt.Exchange:
    cls = getattr(ccxt, EXCHANGE)
    params = {"enableRateLimit": True}
    if API_KEY:
        params["apiKey"] = API_KEY
        params["secret"] = API_SECRET
    ex = cls(params)
    # Sandbox only if keys provided (sandbox requires auth)
    if TESTNET and API_KEY and hasattr(ex, "set_sandbox_mode"):
        ex.set_sandbox_mode(True)
        log.info("Testnet (sandbox) mode active")
    elif not API_KEY:
        log.info("No API keys — public data only (backtest mode)")
    return ex


exchange: ccxt.Exchange = _build_exchange()


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def fetch_ohlcv(limit: int = CANDLES_REQUIRED) -> pd.DataFrame | None:
    """Fetch OHLCV candles and return as DataFrame."""
    try:
        raw = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=limit)
        if not raw:
            return None
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except ccxt.BaseError as exc:
        log.error("fetch_ohlcv failed: %s", exc)
        return None


def fetch_ticker_price() -> float | None:
    """Return the latest mark/last price."""
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        return float(ticker["last"])
    except ccxt.BaseError as exc:
        log.error("fetch_ticker failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

def fetch_balance(quote_currency: str = "USDT") -> float:
    """Return available balance for the quote currency."""
    try:
        bal = exchange.fetch_balance()
        return float(bal["free"].get(quote_currency, 0))
    except ccxt.BaseError as exc:
        log.error("fetch_balance failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def place_market_order(side: str, qty: float) -> dict | None:
    """
    Place a market order.

    Parameters
    ----------
    side : 'buy' or 'sell'
    qty  : base currency quantity
    """
    try:
        order = exchange.create_market_order(SYMBOL, side, qty)
        log.info("Market %s order placed: %s", side.upper(), order["id"])
        return order
    except ccxt.InsufficientFunds as exc:
        log.error("Insufficient funds: %s", exc)
    except ccxt.BaseError as exc:
        log.error("place_market_order failed: %s", exc)
    return None


def place_limit_order(side: str, qty: float, price: float) -> dict | None:
    """Place a limit order."""
    try:
        order = exchange.create_limit_order(SYMBOL, side, qty, price)
        log.info("Limit %s order placed @ %s: %s", side.upper(), price, order["id"])
        return order
    except ccxt.BaseError as exc:
        log.error("place_limit_order failed: %s", exc)
    return None


def cancel_order(order_id: str) -> bool:
    """Cancel an open order."""
    try:
        exchange.cancel_order(order_id, SYMBOL)
        log.info("Order %s cancelled", order_id)
        return True
    except ccxt.OrderNotFound:
        return True   # already gone
    except ccxt.BaseError as exc:
        log.error("cancel_order failed: %s", exc)
        return False


def get_order_status(order_id: str) -> str | None:
    """Return the status string of an order."""
    try:
        order = exchange.fetch_order(order_id, SYMBOL)
        return order["status"]
    except ccxt.BaseError as exc:
        log.error("get_order_status failed: %s", exc)
        return None
