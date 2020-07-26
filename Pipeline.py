import pandas as pd
import numpy as np

__all__ = ["rets", "sector_rets", "market_cap_raw_value", "market_cap", "pe", "pe_lyr", "pb", "ps", "pcf",
           "turnover"]

# read data in
price = pd.read_csv("price.csv")
fundamental = pd.read_csv("fundamental.csv")
sector = pd.read_csv("sector.csv")

# Data cleaning
# Return
if list(sector.columns) != ['date', 'industry', "sector_close"]:
    sector.columns = ['date', 'industry', "sector_close"]

price_sector_merged = price.merge(sector, how='left', on=['date', 'industry'])
price_sector_merged = price_sector_merged.drop('industry', axis=1).set_index("date")
rets = price_sector_merged.groupby(["ticker"]).pct_change()
rets.columns = ['rets', 'sector_returns']
rets["ticker"] = price_sector_merged["ticker"]
sector_rets = rets[['sector_returns', 'ticker']].reset_index().pivot_table(columns='ticker', index='date')
rets = rets[['rets', 'ticker']].reset_index().pivot_table(columns="ticker", index='date')
rets = rets.droplevel(axis=1, level=0)
rets = rets.dropna(how="all", axis=0)

sector_rets = sector_rets.dropna(how="all", axis=0)


# Data wrangling
def zscore(df):
    score = (df - df.mean()) / df.std()

    return score


def winsorize(df, lb=0.05, ub=0.95):
    upper, lower = df.quantile(ub, axis=1), df.quantile(lb, axis=1)

    return df.clip(lower, upper, axis=0)


def fillNA(df, threshold=0.2, limit=5):
    """
    remove the columns with too many missing values
    """

    mask = df.isna().sum() < len(df) * threshold
    tmp = df.loc[:, mask]
    tmp = tmp.fillna(method='ffill', limit=limit)

    return tmp


def preprocess(df, field, **kwargs):
    df_pivot = df[["date", "ticker", field]].pivot_table(columns='ticker', index='date')
    df_pivot = df_pivot.droplevel(axis=1, level=0)
    df_pivot = fillNA(df_pivot, **kwargs)
    df_zscore = zscore(df_pivot)
    df_zscore = winsorize(df_zscore, **kwargs)

    return df_zscore


market_cap_raw_value = fundamental[["date", "ticker", "circulating_market_cap"]]. \
    pivot_table(columns='ticker', index='date').droplevel(axis=1, level=0)
market_cap_raw_value.index = pd.to_datetime(market_cap_raw_value.index, format="%Y-%m-%d")

market_cap = preprocess(fundamental, "circulating_market_cap")
pe = preprocess(fundamental, "pe_ratio")
pe_lyr = preprocess(fundamental, "pe_ratio_lyr")
pb = preprocess(fundamental, "pb_ratio")
ps = preprocess(fundamental, "ps_ratio")
pcf = preprocess(fundamental, "pcf_ratio")
turnover = preprocess(fundamental, "turnover_ratio")


# validate
# make sure every data frame has the same date range and columns


def validate(*args):
    """ *args: factor inputs"""

    rets.index = pd.to_datetime(rets.index, format='%Y-%m-%d')
    sector_rets.index = pd.to_datetime(sector_rets.index, format='%Y-%m-%d')

    assert all(sector_rets.index == rets.index)

    for df in args:
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")

    row_intersection = rets.index
    col_intersection = rets.columns
    for df in args:
        row_intersection = sorted(set(df.index) & set(row_intersection))
        col_intersection = sorted(set(df.columns) & set(col_intersection))

    _rets = rets.loc[row_intersection, col_intersection]
    _sector_rets = sector_rets.loc[row_intersection, col_intersection]

    factor_tuple = tuple([_rets] + [_sector_rets] + [e.loc[row_intersection, col_intersection] for e in args])

    return factor_tuple


rets, sector_rets, market_cap, pe, pe_lyr, pb, ps, pcf, turnover = \
    validate(market_cap, pe, pe_lyr, pb, ps, pcf, turnover)

market_cap_raw_value = market_cap_raw_value.loc[market_cap.index, market_cap.columns]
