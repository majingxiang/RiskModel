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


def one_step_cross_section_fit(ret, **kwargs):
    factor_returns = pd.DataFrame(0, index=ret.index, columns=["const"] + list(kwargs.keys()))  # intercept
    R2 = pd.DataFrame(0, index=ret.index, columns=['R2'])
    t_stats = pd.DataFrame(0, index=ret.index, columns=list(kwargs.keys()))
    residual = []

    for index, value in ret.iterrows():
        X = pd.concat([e.loc[index] for e in kwargs.values()], axis=1)
        X.columns = list(kwargs.keys())
        X = sm.add_constant(X)
        y = value

        ols = sm.OLS(y.values, X, missing="drop").fit()
        r2 = ols.rsquared
        t = ols.tvalues
        factor_ret = ols.params

        intersection = sorted(set(X.dropna().index) & set(y.dropna().index))
        resid = ols.resid
        assert len(intersection) == len(resid)
        residual.append(pd.Series(resid, index=intersection))

        factor_returns.loc[index] = factor_ret
        t_stats.loc[index] = t
        R2.loc[index] = r2

    R2.plot(title="R2")
    t_stats.rolling(22).mean().plot(subplots=True, layout=(4, 2), title="Moving average T stats")

    residual = pd.concat(residual, axis=1)
    residual.columns = ret.index
    residual = residual.T

    return factor_returns, residual


def one_step_cross_section_analyze(ret, coef, **kwargs):
    R2 = pd.DataFrame(0, index=TICKER, columns=["R2"])
    Adj_R2 = pd.DataFrame(0, index=TICKER, columns=["Adj R2"])
    resid_autocorr = pd.DataFrame(0, index=TICKER, columns=["Lag1", "Lag5", "Lag20"])

    for t in TICKER:
        k, v = list(kwargs.keys()), list(kwargs.values())
        x = pd.concat([e[t] for e in v], axis=1)
        x.columns = k
        y = ret[t]

        # remove NA's
        intersection = sorted(list(set(x.dropna().index) & set(y.dropna().index)))

        x, y = x.loc[intersection], y.loc[intersection]
        p = len(k)
        n = len(x)

        y_hat = sm.add_constant(x).dot(coef.values)
        assert len(y_hat) == len(y)

        residual = y_hat.sub(y, axis=0)
        total_ss = np.nansum((y - y.mean()) ** 2)
        residual_ss = np.nansum(residual ** 2)

        r2 = 1 - residual_ss / total_ss
        adj_r2 = 1 - (residual_ss / (n - p - 1)) / (total_ss / n)

        R2.loc[t] = r2
        Adj_R2.loc[t] = adj_r2
        resid_autocorr.loc[t] = [residual.autocorr(1), residual.autocorr(5), residual.autocorr(20)]

    return R2, Adj_R2, resid_autocorr


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y%m%d")
    logger.add("{}_{}.log".format("Cross section regression out-of-sample", now))

    factor_returns, residual = one_step_cross_section_fit(rets_out, market_cap=market_cap, pe=pe, pe_lyr=pe_lyr, pb=pb,
                                                          ps=ps, pcf=pcf, turnover=turnover)

    factor_returns.plot(subplots=True, layout=(4, 2))
    logger.info("Cross section residual autocorrelation")
    logger.info(format_for_print(pd.DataFrame(residual.apply(lambda x: x.autocorr(1)).round(3)).describe()))

    coef = factor_returns.mean()

    R2, adj_R2, resid_autocorr = one_step_cross_section_analyze(rets_out, coef, market_cap=market_cap, pe=pe,
                                                                pe_lyr=pe_lyr, pb=pb,
                                                                ps=ps, pcf=pcf, turnover=turnover)

    logger.info("R2")
    logger.info(format_for_print(R2.describe().round(4)))
    logger.info("Adj R2")
    logger.info(format_for_print(adj_R2.describe().round(4)))
    logger.info("Residual autocorrelation")
    logger.info(format_for_print(resid_autocorr.describe().round(4)))
