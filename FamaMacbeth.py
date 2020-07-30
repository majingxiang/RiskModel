"""
Fama Macbeth regression
1. At each time t, run cross sectional regression to get factor returns
2.

"""

# outside module
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels import PanelOLS, FamaMacBeth
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


def preprocess():
    X = pd.concat(
        [market_cap.stack(), pe.stack(), pe_lyr.stack(), pb.stack(), ps.stack(), pcf.stack(), turnover.stack()],
        axis=1)
    X.columns = ['market_cap', 'pe', 'pe_lyr', 'pb', 'ps', 'pcf', 'turnover']
    X = sm.add_constant(X)
    y = rets_out.stack().loc[X.index]

    X = X.reorder_levels(order=["ticker", "date"])
    y = y.reorder_levels(order=["ticker", "date"])

    data = X.join(pd.DataFrame(y, columns=["ret"]))

    return data


def one_step_panel_fit(data):
    """
    Panel regression is exactly the same as pooled regression!!!
    All coefficients estimation are the same
    """

    fit = PanelOLS(data['ret'], data[['const', 'market_cap', 'pe', 'pe_lyr', 'pb', 'ps', 'pcf', 'turnover']]).fit()

    logger.info("Panel Regression")
    logger.info(fit)
    resid = fit.resids
    logger.info("Residual auto correlation")
    logger.info(format_for_print(pd.DataFrame([resid.autocorr(1), resid.autocorr(5), resid.autocorr(20)])))

    return resid


def one_step_fama_macbeth(data):
    fit = FamaMacBeth(data['ret'], data[['const', 'market_cap', 'pe', 'pe_lyr', 'pb', 'ps', 'pcf', 'turnover']]).fit()

    logger.info("Panel Regression")
    logger.info(fit)
    resid = fit.resids
    logger.info("Residual auto correlation")
    logger.info(format_for_print(pd.DataFrame([resid.autocorr(1), resid.autocorr(5), resid.autocorr(20)],
                                              index=["Lag1", "Lag5", "Lag20"])))

    return resid


if __name__ == "__main__":
    data = preprocess()
    # one_step_panel_fit(data)

    one_step_fama_macbeth(data)
