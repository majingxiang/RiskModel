"""

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
from tqdm import tqdm

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


# todo: fix the residual wiht missing values
def factor_returns(ret, **kwargs):
    """
    Return factor returns and residuals

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
    residual = pd.DataFrame(0, index=ret.index, columns=ret.columns)
    factor_exposure = {}
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
            factor_returns.loc[index] = factor_ret
            residual.loc[index] = resid
            factor_exposure[index] = X
        except:
            logger.warning("OLS catches error at time {}".format(index))
            pass

    return factor_returns, factor_exposure, residual


def factor_cov(factor_exposure, factor_rets, window):
    assert type(factor_exposure) == dict

    factor_raw_cov = factor_rets.rolling(window).cov().dropna(how="all")
    index1 = factor_rets.index[window - 1:]
    index2 = list(factor_exposure.keys())
    intersection = sorted(list(set(index1) & set(index2)))

    result = {}
    for i in intersection:
        _X = factor_exposure[i]
        _cov = factor_raw_cov.loc[i]

        a = np.matmul(_X, _cov, dtype=np.float32)
        b = np.matmul(a.values, _X.T.values, dtype=np.float32)
        c = pd.DataFrame(b, columns=_X.index, index=_X.index)
        result[i] = c

    return result


def residual_cov(residual, window):
    residual_raw_var = residual.rolling(window).var().dropna(how="all")
    return residual_raw_var


def factor_vol(factor_rets, factor_exposure, residual, window):
    """analyze the ewm effect embedded into the model"""

    cov1 = factor_cov(factor_exposure, factor_rets, window)
    cov2 = residual_cov(residual, window)

    intersection = sorted(list(set(cov1.keys()) & set(cov2.index)))

    l = []
    for i in tqdm(intersection):
        _cov = cov1[i] + np.diag(cov2.loc[i])
        tmp = pd.Series(np.sqrt(np.diag(_cov)), index=_cov.columns)
        l.append(tmp)
    factor_raw_vol = pd.concat(l, axis=1)
    factor_raw_vol = factor_raw_vol.T
    factor_raw_vol.index = intersection

    return factor_raw_vol


def Newey_West(factor_exposure, factor_rets, residual, window, name, lag=5):
    """

    Parmas:
        rets: factor return or residual
    """

    def gamma_func(dfr, i=1):
        T = len(dfr.index)
        tmp = np.zeros([len(dfr.columns), len(dfr.columns)])
        for t in range(0, T - i):
            tmp += np.outer(dfr.iloc[t, :], dfr.iloc[t + i, :])
        tmp /= (T - 1)
        return pd.DataFrame(tmp, index=dfr.columns, columns=dfr.columns)

    def adjustment(dfr, q=5):
        g0 = dfr.cov()
        for qi in range(0, q):
            qi += 1
            gi = gamma_func(dfr, qi)
            w = 1 - qi / (1 + q)
            g0 += w * (gi + gi.T)
        return g0

    n = len(factor_rets)
    index = factor_rets.index

    l = []
    for i in tqdm(range(n - window)):
        start, end = index[i], index[i + window-1]
        _factor_cov = adjustment((factor_rets - factor_rets.mean()).loc[start:end, :], q=lag)
        _X = factor_exposure[end]
        a = np.matmul(_X, _factor_cov, dtype=np.float32)
        b = np.matmul(a.values, _X.T.values, dtype=np.float32)
        c = pd.DataFrame(b, columns=_X.index, index=_X.index)
        _residual_var = np.diag(adjustment(residual.iloc[i:i + window, :], q=lag))
        total_cov = c + _residual_var
        tmp = pd.Series(np.sqrt(np.diag(total_cov)), index=total_cov.columns, name=end)
        l.append(tmp)

    factor_raw_vol = pd.concat(l, axis=1)
    factor_raw_vol = factor_raw_vol.T

    factor_raw_vol.to_csv("newey_west_vol_{}d_{}.csv".format(window, name))

    return


def eigen_factor_adjustment():
    return


def VRA():
    """volatility regime adjustment"""
    return


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def cov_to_vol(factor_raw_cov):
    """covariance to volatility"""

    assert type(factor_raw_cov) == dict

    l = []
    for k, v in factor_raw_cov.items():
        l.append(pd.Series(np.sqrt(np.diag(v)), index=v.columns))
    factor_raw_vol = pd.concat(l, axis=1)
    factor_raw_vol = factor_raw_vol.T
    factor_raw_vol.index = list(factor_raw_cov.keys())

    return


if __name__ == "__main__":
    factor_rets, factor_exposure, residual = \
        factor_returns(rets_out, market_cap=market_cap, pe=pe, pe_lyr=pe_lyr, pb=pb, ps=ps, pcf=pcf, turnover=turnover)

    assert list(factor_exposure.keys()) == list(factor_rets.index)

    window = 84

    # factor_ewm_rets = factor_rets.ewm(halflife=window, min_periods=window * 3).mean().dropna(how="all")
    # residual_ewm = residual.ewm(halflife=window, min_periods=window * 3).mean().dropna(how="all")

    # factor_raw_cov = factor_cov(factor_exposure, factor_rets, window)
    # factor_ewm_cov = factor_cov(factor_exposure, factor_ewm_rets, window)
    # factor_raw_vol_63d = cov_to_vol(factor_raw_cov)
    # factor_ewm_vol_63d = cov_to_vol(factor_ewm_cov)
    # real_vol_63d = rets_out.rolling(window).std().dropna(how="all")
    # real_vol_63d = real_vol_63d.loc[factor_raw_vol_63d.index]

    # factor_raw_vol = factor_vol(factor_rets, factor_exposure, residual, window)
    # factor_raw_vol.to_csv("factor_raw_vol_{}d.csv".format(window))
    #
    # factor_ewm_vol = factor_vol(factor_ewm_rets, factor_exposure, residual, window)
    # factor_ewm_vol.to_csv("factor_ewm_vol_{}d.csv".format(window))

    # realize_vol = rets_out.rolling(window).std().dropna(how="all")
    # realize_vol.to_csv("realize_vol_{}d.csv".format(window))
    #
    # from pathos.multiprocessing import Pool
    #
    # pool = Pool(5)
    #
    # f1 = {k: v for i, (k, v) in enumerate(factor_exposure.items()) if i < 300}
    # f2 = {k: v for i, (k, v) in enumerate(factor_exposure.items()) if 300 <= i < 600}
    # f3 = {k: v for i, (k, v) in enumerate(factor_exposure.items()) if 600 <= i < 900}
    # f4 = {k: v for i, (k, v) in enumerate(factor_exposure.items()) if 900 <= i < 1200}
    # f5 = {k: v for i, (k, v) in enumerate(factor_exposure.items()) if i >= 1200}
    #
    # set1 = (f1, factor_rets.iloc[:300], residual.iloc[:300], window, "part1")
    # set2 = (f2, factor_rets.iloc[300:600], residual.iloc[300:600], window, "part2")
    # set3 = (f3, factor_rets.iloc[600:900], residual.iloc[600:900], window, "part3")
    # set4 = (f4, factor_rets.iloc[900:1200], residual.iloc[900:1200], window, "part4")
    # set5 = (f5, factor_rets.iloc[1200:], residual.iloc[1200:], window, "part5")
    #
    # pool.starmap(Newey_West, [set1, set2, set3, set4, set5])
