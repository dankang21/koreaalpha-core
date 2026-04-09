"""Technical analysis indicators — EMA, SMA, RSI, MACD, Bollinger Bands.

Pure computation. No I/O, no external dependencies beyond math.
All functions accept plain Python lists and return lists.
"""

from __future__ import annotations

import math


def ema(values: list[float], period: int) -> list[float | None]:
    """Exponential Moving Average.

    Args:
        values: Price series.
        period: EMA period (window size).

    Returns:
        List of same length; leading entries are None until enough data.
    """
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result
    k = 2 / (period + 1)
    result[period - 1] = sum(values[:period]) / period
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)  # type: ignore
    return result


def sma(values: list[float], period: int) -> list[float | None]:
    """Simple Moving Average.

    Args:
        values: Price series.
        period: SMA period (window size).

    Returns:
        List of same length; leading entries are None until enough data.
    """
    result: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1 : i + 1]) / period
    return result


def rsi(closes: list[float], period: int = 14) -> list[float | None]:
    """Relative Strength Index.

    Args:
        closes: Closing price series.
        period: RSI period (default 14).

    Returns:
        RSI values (0-100). None where insufficient data.
    """
    result: list[float | None] = [None] * len(closes)
    if len(closes) < period + 1:
        return result

    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        result[period] = 100 - 100 / (1 + avg_gain / avg_loss)

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            result[i + 1] = 100 - 100 / (1 + avg_gain / avg_loss)

    return result


def macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> dict[str, list[float | None]]:
    """Moving Average Convergence Divergence.

    Args:
        closes: Closing price series.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal_period: Signal line EMA period (default 9).

    Returns:
        Dict with keys "macd", "signal", "histogram".
    """
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)

    macd_line: list[float | None] = [None] * len(closes)
    for i in range(len(closes)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]

    macd_values = [v for v in macd_line if v is not None]
    sig = ema(macd_values, signal_period) if len(macd_values) >= signal_period else [None] * len(macd_values)

    signal_line: list[float | None] = [None] * len(closes)
    histogram: list[float | None] = [None] * len(closes)
    j = 0
    for i in range(len(closes)):
        if macd_line[i] is not None:
            if j < len(sig) and sig[j] is not None:
                signal_line[i] = sig[j]
                histogram[i] = macd_line[i] - sig[j]
            j += 1

    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def bollinger_bands(
    closes: list[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> dict[str, list[float | None]]:
    """Bollinger Bands.

    Args:
        closes: Closing price series.
        period: SMA period (default 20).
        std_dev: Standard deviation multiplier (default 2.0).

    Returns:
        Dict with keys "middle", "upper", "lower".
    """
    middle = sma(closes, period)
    upper: list[float | None] = [None] * len(closes)
    lower: list[float | None] = [None] * len(closes)

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1 : i + 1]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / (len(window) - 1)
        std = math.sqrt(variance)
        if middle[i] is not None:
            upper[i] = middle[i] + std_dev * std
            lower[i] = middle[i] - std_dev * std

    return {"middle": middle, "upper": upper, "lower": lower}
