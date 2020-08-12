import pandas as pd
import numpy as np

factor_raw_vol = pd.read_csv("C:\\Users\\tjmaj\\Desktop\\RiskModel\\Barra\\factor_raw_vol_84d.csv", index_col=0)
factor_nw_vol = pd.read_csv("C:\\Users\\tjmaj\\Desktop\\RiskModel\\Barra\\newey_west_vol_84d.csv", index_col=0)
real_vol = pd.read_csv("C:\\Users\\tjmaj\\Desktop\\RiskModel\\Barra\\realize_vol_84d.csv", index_col=0)
factor_nw_vol.index = pd.to_datetime(factor_nw_vol.index)
factor_raw_vol.index = pd.to_datetime(factor_raw_vol.index)
real_vol.index = pd.to_datetime(real_vol.index)

factor_raw_vol = factor_raw_vol * np.sqrt(252)
factor_nw_vol = factor_nw_vol * np.sqrt(252)
real_vol = real_vol * np.sqrt(252)

idx = np.random.randint(0, len(factor_nw_vol.columns), 10)
for i in idx:
    tmp = pd.concat([factor_raw_vol.iloc[:, i], factor_nw_vol.iloc[:, i], real_vol.iloc[:, i]], axis=1)
    tmp.columns = ["factor vol", "newey west adjust vol", "realize"]
    tmp.plot(title=factor_raw_vol.columns[i])
