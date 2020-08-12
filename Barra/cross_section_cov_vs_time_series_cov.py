"""
Compare the cross sectional cov and time series cov

"""

# outside module
import numpy as np
import pandas as pd
import statsmodels.api as sm
import linearmodels
from loguru import logger
import matplotlib.pyplot as plt
import datetime
from pathlib import Path

# inside module
from Utils import *
from Pipeline import rets, sector_rets, market_cap_raw_value, market_cap, pe, pe_lyr, pb, ps, pcf, turnover, dummy

_tmp = abs(rets).max()
mask = _tmp[_tmp < 0.11].index

rets, sector_rets, market_cap, pe, pe_lyr, pb, ps, pcf, turnover, dummy = \
    rets[mask], sector_rets[mask], market_cap[mask], pe[mask], pe_lyr[mask], pb[mask], ps[mask], pcf[mask], \
    turnover[mask], dummy.loc[mask]

TICKER = list(rets.columns)
rets_out = rets.shift(-1).dropna(how="all")
weights_out = np.sqrt(market_cap_raw_value.loc[rets_out.index])
weights_out = weights_out[mask]


def dummy_fit(ret, **kwargs):
    """
    We calculate the factor returns and covariance for every time t through cross sectional regression

    Dummy means we treat each sector return as a factor, the factor exposure is 1 if that equity in
    the industry, otherwise 0

    X:
        intercept: country factor
        industry dummy: one hot
        style factor: pe (zscore) etc

    """
    col_name = ["const"] + list(dummy.columns) + list(kwargs.keys())
    factor_returns = \
        pd.DataFrame(0, index=ret.index, columns=col_name)  # intercept
    covs = pd.DataFrame(0, index=pd.MultiIndex.from_product([list(ret.index), col_name]),
                        columns=col_name)
    residual = pd.DataFrame(0, index=ret.index, columns=["Residual"])
    for index, value in ret.iterrows():

        _X = pd.concat([e.loc[index] for e in kwargs.values()], axis=1)
        _X.columns = list(kwargs.keys())

        assert _X.shape[0] == dummy.shape[0]

        X = sm.add_constant(pd.concat([dummy, _X], axis=1))
        y = value

        try:
            ols = sm.WLS(y.values, X, missing="drop", weights=weights_out.loc[index]).fit()
            resid = ols.resid
            factor_ret = ols.params
            cov = ols.cov_HC0
            factor_returns.loc[index] = factor_ret
            covs.loc[index] = cov
            residual.loc[index] = resid
        except:
            logger.warning("OLS catches error at time {}".format(index))
            pass

    return factor_returns, residual, covs


def empirical_cov(rets, window):
    cov = rets.rolling(window).cov().dropna(how="all")
    index = pd.MultiIndex.from_product([rets.index[window - 1:], rets.columns])
    cov.index = index
    return cov


def cov_to_vol(x):
    vol = pd.DataFrame(0, index=x.index.levels[0], columns=x.columns)
    for index in x.index.levels[0]:
        series = x.loc[index]
        vol.loc[index] = np.sqrt(np.diag(series))

    return vol


if __name__ == "__main__":
    # cross sectional cov
    factor_returns, residual, cs_cov = dummy_fit(rets_out, market_cap=market_cap, pe=pe, pe_lyr=pe_lyr, pb=pb,
                                                 ps=ps, pcf=pcf, turnover=turnover)

    # time series cov
    factor_returns.index.name = None
    ts_cov_raw = empirical_cov(factor_returns, 63)

    # realize vol
    realize_63d_vol = factor_returns.rolling(63).std().dropna(how="all")
    cs_vol = cov_to_vol(cs_cov)
    ts_vol = cov_to_vol(ts_cov_raw)

    # visualization
    assert all(realize_63d_vol.columns == cs_vol.columns) and all(cs_vol.columns == ts_vol.columns)

    for c in realize_63d_vol.columns:
        tmp = pd.concat([realize_63d_vol[c], cs_vol[c], ts_vol[c]], axis=1)
        tmp.columns = ["realize 63d", "cross-section", "time series"]
        tmp.plot(title=c)


