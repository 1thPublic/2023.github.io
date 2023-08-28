# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:38:30 2022

@author: windows
"""

#设置工作路径
import os

os.chdir("D:\文件柜\研究\港股互联网\数据处理")

import pandas as pd

Closedata = pd.read_csv("Close.csv")
print("Data read successfully!")
print("Number of data points:", len(Closedata))

#读入的数据中存在NaN，需要进行遗失值补差
import numpy as np
from sklearn.impute import SimpleImputer

HSI = Closedata["HSI"].values.reshape(-1,1)
HSIII = Closedata["HSIII"].values.reshape(-1,1)
Tencent = Closedata["Tencent"].values.reshape(-1,1)
SMIC = Closedata["SMIC"].values.reshape(-1,1)

Close_imputer = SimpleImputer(missing_values=np.nan, strategy="median")

Closedata["HSI"] = Close_imputer.fit_transform(HSI)
Closedata["HSIII"] = Close_imputer.fit_transform(HSIII)
Closedata["Tencent"] = Close_imputer.fit_transform(Tencent)
Closedata["SMIC"] = Close_imputer.fit_transform(SMIC)

print("Data is successfully processed!")

from pandas_profiling import ProfileReport

prof = ProfileReport(Closedata)
prof.to_file(output_file='report.html')