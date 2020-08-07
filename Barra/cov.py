"""
Cross section regression
1. At each time t, run cross sectional regression to get factor returns
2.

"""

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

TICKER = list(rets.columns)

rets_out = rets.shift(-1).dropna(how="all")
# under this design, the factor return will be based on time t, which is not true
# the factor return dataframe should shift the index by lag1
rets_in = rets

weights_in = np.sqrt(market_cap_raw_value.loc[rets_in.index])
weights_out = np.sqrt(market_cap_raw_value.loc[rets_out.index])


def factor_return():
    return


def residual():
    return


def corr():
    return


def vol():
    return
