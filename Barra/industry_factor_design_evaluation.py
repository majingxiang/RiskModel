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
from Pipeline import rets, sector_rets, market_cap_raw_value, market_cap, pe, pe_lyr, pb, ps, pcf, turnover

"""
We test cross sectional regression with 2 different designs
1. use raw sector return as a factor
2. introduce all industries as factors, i.e. dummy variables

"""

_tmp = abs(sector_rets).max()
mask = _tmp[_tmp < 0.1].index

rets, sector_rets, market_cap, pe, pe_lyr, pb, ps, pcf, turnover = \
    rets[mask], sector_rets[mask], market_cap[mask], pe[mask], pe_lyr[mask], pb[mask], ps[mask], pcf[mask], \
    turnover[mask]

TICKER = list(rets.columns)
rets_out = rets.shift(-1).dropna(how="all")
weights_out = np.sqrt(market_cap_raw_value.loc[rets_out.index])


def regular_fit(ret, **kwargs):
    """
    sector return as a single factor

    """
    factor_returns = pd.DataFrame(0, index=ret.index, columns=["const"] + list(kwargs.keys()))  # intercept
    R2 = pd.DataFrame(0, index=ret.index, columns=['R2'])
    adj_R2 = pd.DataFrame(0, index=ret.index, columns=["Adj R2"])
    t_stats = pd.DataFrame(0, index=ret.index, columns=list(kwargs.keys()))
    residual = []

    for index, value in ret.iterrows():
        X = pd.concat([e.loc[index] for e in kwargs.values()], axis=1)
        X.columns = list(kwargs.keys())
        X = sm.add_constant(X)
        y = value

        try:
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
            adj_R2.loc[index] = ols.rsquared_adj
        except:
            logger.warning("OLS catches error at time {}".format(index))
            pass

    residual = pd.concat(residual, axis=1)
    residual.columns = ret.index
    residual = residual.T

    return factor_returns, R2, adj_R2, residual


def dummy_fit(ret, **kwargs):
    """
    Dummy means we treat each sector return as a factor, the factor exposure is 1 if that equity in
    the industry, otherwise 0

    X:
        intercept: country factor
        industry dummy: one hot
        style factor: pe (zscore) etc

    """

    # adjust sector return first
    PATH = str(Path().absolute().parent) + "\\Data"
    price = pd.read_csv(PATH + "\\price.csv")
    sector = pd.read_csv(PATH + "\\sector.csv")
    # Return
    if list(sector.columns) != ['date', 'industry', "sector_close"]:
        sector.columns = ['date', 'industry', "sector_close"]
    price_sector_merged = price.merge(sector, how='left', on=['date', 'industry'])
    price_sector_merged = price_sector_merged[['ticker', 'industry']].drop_duplicates()

    dummy = pd.get_dummies(price_sector_merged, columns=["industry"])
    dummy = dummy.set_index("ticker")
    dummy = dummy.loc[rets.columns]

    factor_returns = pd.DataFrame(0, index=ret.index, columns=list(dummy.columns) + list(kwargs.keys()))  # intercept
    R2 = pd.DataFrame(0, index=ret.index, columns=['R2'])
    adj_R2 = pd.DataFrame(0, index=ret.index, columns=["Adj R2"])
    t_stats = pd.DataFrame(0, index=ret.index, columns=list(dummy.columns) + list(kwargs.keys()))
    residual = []

    for index, value in ret.iterrows():

        # todo: not sure if we need add constant here
        _X = pd.concat([e.loc[index] for e in kwargs.values()], axis=1)
        _X.columns = list(kwargs.keys())

        assert _X.shape[0] == dummy.shape[0]

        X = sm.add_constant(pd.concat([dummy, _X], axis=1))
        y = value

        try:
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
            adj_R2.loc[index] = ols.rsquared_adj
        except:
            logger.warning("OLS catches error at time {}".format(index))
            pass

    residual = pd.concat(residual, axis=1)
    residual.columns = ret.index
    residual = residual.T

    return factor_returns, R2, adj_R2, residual


def analyze():
    regular_factor_returns1, regular_R2_1, regular_adj_R2_1, regular_residual1 = \
        regular_fit(rets_out, market_cap=market_cap, pe=pe, pe_lyr=pe_lyr, pb=pb,
                    ps=ps, pcf=pcf, turnover=turnover)

    regular_factor_returns2, regular_R2_2, regular_adj_R2_2, regular_residual2 = \
        regular_fit(rets_out, sector_rets=sector_rets, market_cap=market_cap, pe=pe, pe_lyr=pe_lyr, pb=pb,
                    ps=ps, pcf=pcf, turnover=turnover)

    dummy_factor_returns, dummy_R2, dummy_adj_R2, dummy_residual = \
        dummy_fit(rets_out, market_cap=market_cap, pe=pe, pe_lyr=pe_lyr, pb=pb,
                  ps=ps, pcf=pcf, turnover=turnover)

    logger.info("Different sector factor construction")

    logger.info("R2 without sector return factor")
    logger.info(format_for_print(pd.DataFrame(regular_R2_1.describe().round(3))))
    logger.info("R2 with sector return factor")
    logger.info(format_for_print(pd.DataFrame(regular_R2_2.describe().round(3))))
    logger.info("R2 with sector dummy variables")
    logger.info(format_for_print(pd.DataFrame(dummy_R2.describe().round(3))))

    (dummy_R2 - regular_R2_2).plot(title="R2 diff between dummy and numeric")
    (dummy_adj_R2 - regular_adj_R2_2).plot(title="Adj R2 diff between dummy and numeric")

    regular_ar = regular_residual2.apply(lambda x: x.autocorr(1))
    dummy_ar = dummy_residual.apply(lambda x: x.autocorr(1))
    logger.info("(dummy residual autocorrelaiton - regular residual autocorrelaiton) /dummy residual autocorrelaiton")
    logger.info(format_for_print(pd.DataFrame((dummy_ar - regular_ar) / regular_ar).describe()).round(3))

    return


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y%m%d")
    logger.add("{}_{}.log".format("Different industry factor construction", now))
    analyze()
