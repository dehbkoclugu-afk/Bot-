"""
Risk Yöneticisi
===============
  Stop-Loss   : giriş − 1.5 × ATR   (long)
  Take-Profit : giriş + 3.5 × ATR   (long)  → 2.3:1 RR
  Pozisyon boyutu: bakiyenin %1'i risk edecek şekilde
"""

from config import RISK_PER_TRADE, ATR_SL_MULT, ATR_TP_MULT


def calculate_levels(entry: float, atr: float, side: str) -> dict:
    sl_dist = ATR_SL_MULT * atr
    tp_dist = ATR_TP_MULT * atr

    if side == "buy":
        stop_loss   = entry - sl_dist
        take_profit = entry + tp_dist
    else:
        stop_loss   = entry + sl_dist
        take_profit = entry - tp_dist

    rr = tp_dist / sl_dist if sl_dist > 0 else 0

    return {
        "take_profit": round(take_profit, 8),
        "stop_loss"  : round(stop_loss,   8),
        "rr_ratio"   : round(rr, 2),
    }


def calculate_position_size(balance: float, entry: float, stop_loss: float) -> float:
    sl_distance = abs(entry - stop_loss)
    if sl_distance == 0:
        return 0.0
    risk_amount = balance * RISK_PER_TRADE
    return round(risk_amount / sl_distance, 8)


def check_exit(
    side: str,
    entry: float,
    current_price: float,
    take_profit: float,
    stop_loss: float,
) -> str:
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
