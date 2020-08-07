# outside module
import random
import numpy as np
import pandas as pd
import statsmodels.api as sm
import linearmodels
from loguru import logger
import matplotlib.pyplot as plt
import datetime
from jqdatasdk import auth, is_auth, get_price

auth("18622056381", "1510103Yhw")
logger.info("Connected to JQData: {}".format(is_auth()))

# inside module
from Utils import format_for_print
from Pipeline import rets, sector_rets, market_cap_raw_value, market_cap, pe, pe_lyr, pb, ps, pcf, turnover

# In sample fit
X_in = pd.concat([market_cap.stack(), pe.stack(), pe_lyr.stack(), pb.stack(), ps.stack(), pcf.stack(),
                  turnover.stack()], axis=1)
X_in.columns = ["market_cap", "pe", "pe_lyr", "pb", "ps", "pcf", "turnover"]
y_in = rets.stack()
y_in = y_in.loc[X_in.index]

y_out = rets.shift(-1).stack()
X_out = X_in.loc[y_out.index]

weights_in = np.sqrt(market_cap_raw_value.stack().loc[X_in.index])
weights_out = np.sqrt(market_cap_raw_value.stack().loc[X_out.index])

assert len(X_out) == len(y_out)

rets_out = rets.shift(-1)
market_price = get_price("000906.XSHG", fields=["close"], start_date='2014-01-01', end_date='2020-01-01')
market_ret = market_price.pct_change()
market_ret_out = market_ret.shift(-1)


def get_residual(rets, market_ret):
    """Get residual from regression with market return"""
    residual = []
    for index, value in rets.iteritems():
        y = value
        X = sm.add_constant(market_ret)

        intersection = sorted(list(set(X.dropna().index) & set(y.dropna().index)))

        fit = sm.OLS(y.values, X, missing="drop").fit()
        resid = fit.resid

        assert len(intersection) == len(resid)
        residual.append(pd.Series(resid, index=intersection))

    result = pd.concat(residual, axis=1)
    result.columns = rets.columns
    return result


resid = get_residual(rets_out, market_ret)
logger.info("Residual from removing market return")
logger.info("Lag 1 auto correlation: {}".format(resid.stack().autocorr(1)))


def reorder(ret, num=20):
    """
    Reorder the columns, then stack the dataframe
    Hopefully, this will reduce some autocorrelation
    """
    for i in range(num):
        col = list(rets.columns)
        random.shuffle(col)
        logger.info("Auto corr for difference order {}".format(ret[col].stack().autocorr()))

    return


def analyze(X_in, y_in, weights):
    # 1. autocorrelation

    autocorr_y_in = pd.DataFrame([y_in.autocorr(1), y_in.autocorr(5), y_in.autocorr(20)]).T.apply(
        lambda x: x.round(3))
    autocorr_y_in.columns = ['Lag1', 'Lag5', 'Lag20']
    autocorr_X_in = pd.concat(
        [X_in.apply(lambda x: x.autocorr(1)), X_in.apply(lambda x: x.autocorr(5)),
         X_in.apply(lambda x: x.autocorr(20))], axis=1).apply(lambda x: x.round(3))
    autocorr_X_in.columns = ['Lag1', 'Lag5', 'Lag20']

    # 2. linear model
    logger.info("OLS")
    ols = sm.OLS(y_in.values, sm.add_constant(X_in), missing="drop").fit()
    logger.info(ols.summary())

    logger.info("WLS")

    wls = sm.WLS(y_in.values, sm.add_constant(X_in), weights=weights,
                 missing="drop").fit()
    logger.info(wls.summary())

    logger.info("Residual standard error")
    bse = pd.concat([wls.bse, ols.bse], axis=1)
    bse.columns = ["WLS (bps)", "OLS (bps)"]
    bse = (bse * 10000).apply(lambda x: x.round(3))
    logger.info(format_for_print(bse))

    # residual autocorrelation

    ols_residual = ols.resid
    wls_residual = wls.resid
    autocorr_ols_resid = pd.DataFrame([ols_residual.autocorr(1), ols_residual.autocorr(5), ols_residual.autocorr(20)]).T \
        .apply(lambda x: x.round(3))
    autocorr_ols_resid.columns = ['Lag1', 'Lag5', 'Lag20']

    autocorr_wls_resid = pd.DataFrame([wls_residual.autocorr(1), wls_residual.autocorr(5), wls_residual.autocorr(20)]).T \
        .apply(lambda x: x.round(3))
    autocorr_wls_resid.columns = ['Lag1', 'Lag5', 'Lag20']

    logger.info("X auto correlation")
    logger.info(format_for_print(autocorr_X_in))

    logger.info("Y auto correlation")
    logger.info(format_for_print(autocorr_y_in))

    logger.info("OLS Residual autocorrelation")
    logger.info(format_for_print(autocorr_ols_resid))

    logger.info("WLS Residual autocorrelation")
    logger.info(format_for_print(autocorr_wls_resid))


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y%m%d")

    # logger.add("{}_{}.log".format("Pooled regression insample", now))
    # analyze(X_in, y_in, weights_in)

    logger.add("{}_{}.log".format("Pooled regression out-of-sample", now))
    analyze(X_out, y_out, weights_out)
