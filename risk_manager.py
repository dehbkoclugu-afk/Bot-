"""
Risk Manager
------------
* Position sizing : risk a fixed % of balance per trade
* Take-profit     : entry + ATR_TP_MULT × ATR
* Stop-loss       : entry - ATR_SL_MULT × ATR
* Trailing-stop   : optional, enabled when price moves > 1 ATR in our favour
"""

from config import RISK_PER_TRADE, ATR_SL_MULT, ATR_TP_MULT


def calculate_levels(entry: float, atr: float, side: str) -> dict:
    """
    Calculate TP / SL prices for a trade.

    Parameters
    ----------
    entry : entry price
    atr   : ATR value at entry candle
    side  : 'buy' or 'sell'

    Returns
    -------
    dict  : { take_profit, stop_loss, rr_ratio }
    """
    sl_distance = ATR_SL_MULT * atr
    tp_distance = ATR_TP_MULT * atr

    if side == "buy":
        stop_loss   = entry - sl_distance
        take_profit = entry + tp_distance
    else:
        stop_loss   = entry + sl_distance
        take_profit = entry - tp_distance

    rr = tp_distance / sl_distance if sl_distance > 0 else 0

    return {
        "take_profit": round(take_profit, 8),
        "stop_loss"  : round(stop_loss,   8),
        "rr_ratio"   : round(rr, 2),
    }


def calculate_position_size(balance: float, entry: float, stop_loss: float) -> float:
    """
    Risk-based position sizing.

    risk_amount  = balance × RISK_PER_TRADE
    position_qty = risk_amount / |entry - stop_loss|
    """
    sl_distance = abs(entry - stop_loss)
    if sl_distance == 0:
        return 0.0

    risk_amount = balance * RISK_PER_TRADE
    qty = risk_amount / sl_distance
    return round(qty, 8)


def check_exit(
    side: str,
    entry: float,
    current_price: float,
    take_profit: float,
    stop_loss: float,
) -> str:
    """
    Evaluate whether an open position should be closed.

    Returns
    -------
    'take_profit' | 'stop_loss' | 'hold'
    """
    if side == "buy":
        if current_price >= take_profit:
            return "take_profit"
        if current_price <= stop_loss:
            return "stop_loss"
    elif side == "sell":
        if current_price <= take_profit:
            return "take_profit"
        if current_price >= stop_loss:
            return "stop_loss"
    return "hold"
