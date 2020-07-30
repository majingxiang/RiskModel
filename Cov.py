# outside module
import numpy as np
import pandas as pd
import statsmodels.api as sm
import linearmodels
from loguru import logger
import matplotlib.pyplot as plt
import datetime

# inside module
from Utils import *
from Pipeline import rets, sector_rets, market_cap_raw_value, market_cap, pe, pe_lyr, pb, ps, pcf, turnover

__all__ = ["empirical_cov", "mixed_cov", "barra_cov"]


def empirical_cov(rets, window):
    cov = rets.rolling(window).cov().dropna(how="all")

    return cov


def mixed_cov(rets, *args):
    """

    Params:
        rets:
        args: window1, window2, window3
    """

    return


def barra_cov(rets, **kwargs):
    """Barra covariance"""

    return


if __name__ == "__main__":
    pass
