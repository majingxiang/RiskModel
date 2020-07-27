"""
One step time series regression:

Time series regression to get coefficient first, then average the coefficient for all equities,
finally only 1 regression line

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

rets_out = rets.shift(-1)
rets_in = rets


def sample_analysis(rets, num):
    """
    Random sample some equity, perform some linear regression test

    1. residual correlation
    2. constant variance, wls
    3. covariance matrix
    3. residual normality
    4. R square, other evaluations
    """

    ticker = [TICKER[e] for e in np.random.randint(0, len(TICKER) - 1, num)]

    for t in ticker:
        logger.info("Security is: {}".format(t))
        X = pd.concat([market_cap[t], pe[t], pe_lyr[t], pb[t], ps[t], pcf[t], turnover[t]], axis=1)
        X.columns = ["market_cap", "pe", "pe_lyr", "pb", "ps", "pcf", "turnover"]
        y = rets[t]
        ols = sm.OLS(y.values, sm.add_constant(X), missing="drop", ).fit()
        logger.info(ols.summary())

        ols_residual = ols.resid
        autocorr_ols_resid = pd.DataFrame(
            [ols_residual.autocorr(1), ols_residual.autocorr(5), ols_residual.autocorr(20)]).T \
            .apply(lambda x: x.round(3))
        autocorr_ols_resid.columns = ['Lag1', 'Lag5', 'Lag20']

        logger.info("Covariance (bps)")
        logger.info(format_for_print(pd.DataFrame(ols.cov_HC0).apply(lambda x: (10000 * x).round(3))))

        logger.info("Residual autocorrelation")
        logger.info(format_for_print(autocorr_ols_resid))

    return


def one_step_time_series_fit(y, **kwargs):
    """

    WLS issue is: unknown weight

    Params:
        y: return DataFrame
        args: factors
    """

    beta = pd.DataFrame()  # Beta for all equity, factor exposure
    for t in TICKER:
        k, v = list(kwargs.keys()), list(kwargs.values())
        x = pd.concat([e[t] for e in v], axis=1)
        x.columns = k
        ols = sm.OLS(y[t].values, sm.add_constant(x), missing="drop").fit()
        beta[t] = pd.Series(ols.params)

    coef = pd.DataFrame(beta.mean(axis=1), columns=["Coef"])  # only support pd.DataFrame now, don;t support pd.Series
    logger.info("Mean Coefficient for all assets")
    logger.info(format_for_print(coef.apply(lambda x: x.round(5))))

    se = pd.DataFrame(beta.std(axis=1), columns=["Standard error"])
    logger.info("Standard error of coefficient")
    logger.info(format_for_print(se.apply(lambda x: x.round(5))))

    t_stats = coef.div(se.values, axis=0)
    t_stats.columns = ["t-stats"]
    logger.info("Coef T-stats")
    logger.info(format_for_print(t_stats.apply(lambda x: x.round(3))))

    return beta, coef


def one_step_time_series_analyze(ret, coef, **kwargs):
    """deal with missing values"""

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

        residual = y_hat.sub(y, axis=0).iloc[:, 0]
        total_ss = np.nansum((y - y.mean()) ** 2)
        residual_ss = np.nansum(y_hat.sub(y, axis=0) ** 2)

        r2 = 1 - residual_ss / total_ss
        adj_r2 = 1 - (residual_ss / (n - p - 1)) / (total_ss / n)

        R2.loc[t] = r2
        Adj_R2.loc[t] = adj_r2
        resid_autocorr.loc[t] = [residual.autocorr(1), residual.autocorr(5), residual.autocorr(20)]

    return R2, Adj_R2, resid_autocorr


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y%m%d")
    logger.add("{}_{}.log".format("Time series regression out-of-sample", now))

    sample_analysis(rets_out, 10)

    factor_exposure, coef = \
        one_step_time_series_fit(rets_out, market_cap=market_cap, pe=pe, pe_lyr=pe_lyr, pb=pb, ps=ps, pcf=pcf,
                                 turnover=turnover)

    R2, Adj_R2, residual_autocorr = \
        one_step_time_series_analyze(rets_out, coef, market_cap=market_cap, pe=pe, pe_lyr=pe_lyr, pb=pb, ps=ps,
                                     pcf=pcf, turnover=turnover)

    logger.info("R2")
    logger.info(format_for_print(R2.describe().round(4)))
    logger.info("Adj R2")
    logger.info(format_for_print(Adj_R2.describe().round(4)))
    logger.info("Residual autocorrelation")
    logger.info(format_for_print(residual_autocorr.describe().round(4)))
