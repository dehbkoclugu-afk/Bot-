"""
Risk Yöneticisi v3
==================
  Başlangıç SL  : giriş ± 1.2 × ATR
  Trailing Stop : +2.0×ATR'de breakeven; +3.0×ATR'de peak izleme (peak − 1.5×ATR)
  Emniyet kapağı: giriş + 5.5 × ATR
  Çıkış: cap_tp | trail_sl | cross_exit (bot.py'de MA çapraz kontrolü)
  Pozisyon boyutu: bakiyenin %1'i riske edilir
"""

from config import (
    RISK_PER_TRADE,
    ATR_SL_MULT,
    ATR_TRAIL_BREAKEVEN,
    ATR_TRAIL_ACTIVATE,
    ATR_TRAIL_DIST,
    ATR_CAP_MULT,
)


def calculate_levels(entry: float, atr: float, side: str) -> dict:
    """
    Başlangıç SL ve emniyet TP kapağını hesaplar.
    Gerçek çıkış trailing stop ile yapılır; cap_tp yalnızca emniyet valfidir.
    """
    sl_dist  = ATR_SL_MULT * atr
    cap_dist = ATR_CAP_MULT * atr

    if side == "buy":
        stop_loss = entry - sl_dist
        cap_tp    = entry + cap_dist
    else:
        stop_loss = entry + sl_dist
        cap_tp    = entry - cap_dist

    return {
        "stop_loss": round(stop_loss, 8),
        "cap_tp"   : round(cap_tp,   8),
        "rr_ratio" : ATR_CAP_MULT / ATR_SL_MULT,   # teorik maks RR
    }


def calculate_position_size(balance: float, entry: float, stop_loss: float,
                             trade_risk: float = None) -> float:
    sl_distance = abs(entry - stop_loss)
    if sl_distance == 0:
        return 0.0
    risk = trade_risk if trade_risk is not None else RISK_PER_TRADE
    return round(balance * risk / sl_distance, 8)


def update_trailing_stop(
    side: str,
    entry: float,
    atr: float,
    current_price: float,
    current_sl: float,
    extreme: float,       # buy için: şimdiye kadar görülen en yüksek; sell için en düşük
) -> tuple[float, float]:
    """
    Her bar çağrılır. SL'yi ve extreme değerini günceller.

    Returns
    -------
    (new_sl, new_extreme)
    Buy tarafında SL yalnızca yukarı gider (asla düşmez).
    """
    if side == "buy":
        new_extreme = max(extreme, current_price)
        new_sl      = current_sl

        move = new_extreme - entry

        if move >= ATR_TRAIL_ACTIVATE * atr:
            # Trailing aktif: peak'in 1.5×ATR altını takip et
            trail_sl = new_extreme - ATR_TRAIL_DIST * atr
            new_sl = max(current_sl, trail_sl)
        elif move >= ATR_TRAIL_BREAKEVEN * atr:
            # Breakeven: SL'yi giriş fiyatına çek
            new_sl = max(current_sl, entry)

        return new_sl, new_extreme

    else:  # sell
        new_extreme = min(extreme, current_price)
        new_sl      = current_sl

        move = entry - new_extreme

        if move >= ATR_TRAIL_ACTIVATE * atr:
            trail_sl = new_extreme + ATR_TRAIL_DIST * atr
            new_sl = min(current_sl, trail_sl)
        elif move >= ATR_TRAIL_BREAKEVEN * atr:
            new_sl = min(current_sl, entry)

        return new_sl, new_extreme


def check_exit(
    side: str,
    current_price: float,
    stop_loss: float,
    cap_tp: float,
) -> str:
    """'cap_tp' | 'trail_sl' | 'hold'"""
    if side == "buy":
        if current_price >= cap_tp:    return "cap_tp"
        if current_price <= stop_loss: return "trail_sl"
    elif side == "sell":
        if current_price <= cap_tp:    return "cap_tp"
        if current_price >= stop_loss: return "trail_sl"
    return "hold"
