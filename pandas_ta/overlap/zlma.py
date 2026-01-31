# -*- coding: utf-8 -*-
from sys import modules as sys_modules
from numpy import isnan
from pandas import Series
from pandas_ta._typing import DictLike, Int
from pandas_ta.utils import v_mamode, v_offset, v_pos_default, v_series

# Available MAs for zlma


def zlma(
    close: Series,
    length: Int = None,
    mamode: str = None,
    offset: Int = None,
    **kwargs: DictLike,
) -> Series:
    """Zero Lag Moving Average (ZLMA)

    The Zero Lag Moving Average attempts to eliminate the lag associated
    with moving averages. This is an adaption created by John Ehler
    and Ric Way.

    Sources:
        https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average

    Args:
        close (pd.Series): Series of 'close's
        length (int): It's period. Default: 10
        mamode (str): Options: 'ema', 'hma', 'sma', 'wma'. Default: 'ema'
        offset (int): How many periods to offset the result. Default: 0

    Kwargs:
        fillna (value, optional): pd.DataFrame.fillna(value)

    Returns:
        pd.Series: New feature generated.
    """
    # Validate
    length = v_pos_default(length, 10)
    close = v_series(close, length)

    if close is None:
        return

    mamode = v_mamode(mamode, "ema")
    supported_mas = [
        "dema",
        "ema",
        "fwma",
        "hma",
        "linreg",
        "midpoint",
        "pwma",
        "rma",
        "sinwma",
        "sma",
        "ssf",
        "swma",
        "t3",
        "tema",
        "trima",
        "vidya",
        "wma",
    ]

    if mamode not in supported_mas:
        return

    offset = v_offset(offset)

    # Calculate
    lag = int(0.5 * (length - 1))
    close_ = 2 * close - close.shift(lag)

    kwargs.update({"close": close_})
    kwargs.update({"length": length})

    fn = getattr(sys_modules[__name__], mamode)
    zlma = fn(**kwargs)

    if zlma is None or all(isnan(zlma)):
        return  # Emergency Break

    # Offset
    if offset != 0:
        zlma = zlma.shift(offset)

    # Fill
    if "fillna" in kwargs:
        zlma = zlma.fillna(kwargs["fillna"])

    # Name and Category
    zlma.name = f"ZL_{zlma.name}"
    zlma.category = "overlap"

    return zlma
