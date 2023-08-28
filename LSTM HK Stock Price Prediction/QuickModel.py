# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:34:30 2022

@author: windows
"""
import os

os.chdir("D:\文件柜\研究\港股互联网\数据处理")

# 这是用来做股指、股票数据处理的代码，目标效果如论文所示
# 首先将三组数据导入，按照时间序列对数据进行处理，使用循环语句算出所需9组指标
# 其次，将数据打散为独立，从每年的数据中抽取占样本总量20%的天数作为训练集和检测集
# 读取10年间HSI, HSIII指数以及腾讯和中芯国际的股价数据
# 数据源: Choice金融终端
import pandas as pd

HSIdata = pd.read_csv("SSE.600000.csv")

print("Data read successfully!")
print("Number of data points:", len(HSIdata))

#读入的数据中存在NaN，需要进行遗失值补差
import numpy as np
from sklearn.impute import SimpleImputer

Open = HSIdata["Open"].values.reshape(-1,1)
Low = HSIdata["Low"].values.reshape(-1,1)
High = HSIdata["High"].values.reshape(-1,1)
Close = HSIdata["Close"].values.reshape(-1,1)
Movement = HSIdata["Movement"].values.reshape(-1,1)
Volume = HSIdata["Volume"].values.reshape(-1,1)
Turnover = HSIdata["Turnover"].values.reshape(-1,1)

print("!")

HSI_imputer = SimpleImputer(missing_values=np.nan, strategy="median")

HSIdata["Open"] = HSI_imputer.fit_transform(Open)
HSIdata["Low"] = HSI_imputer.fit_transform(Low)
HSIdata["High"] = HSI_imputer.fit_transform(High)
HSIdata["Close"] = HSI_imputer.fit_transform(Close)
HSIdata["Movement"] = HSI_imputer.fit_transform(Movement)
HSIdata["Volume"] = HSI_imputer.fit_transform(Volume)
HSIdata["Turnover"] = HSI_imputer.fit_transform(Turnover)

#将Movement换成涨跌的标签，以0表示跌，1表示涨
MovFilter = lambda x : 1 if x > 0 else 0
HSIdata["Movement"] = list(map(MovFilter, HSIdata["Movement"].tolist()))

print("!")

#在对数据进行抽样之前，计算各个时间序列指标数据
import talib as tl

HSIdata["Date"] = pd.to_datetime(HSIdata["Date"])# 将“交易时间”从string转化成datetime，便于数据处理
HSIdata["Volume"] = HSIdata["Turnover"]/HSIdata["Close"] 
HSIdata["SMA"] = tl.SMA(HSIdata["Close"], timeperiod = 10)
HSIdata["WMA"] = tl.WMA(HSIdata["Close"], timeperiod = 10)
HSIdata["Momentum"] = tl.MOM(HSIdata["Close"], timeperiod = 9)
HSIdata["K"], HSIdata["D"] = tl.STOCHF(HSIdata["High"], HSIdata["Low"], HSIdata["Close"], 
                                       fastk_period=5, fastd_period=3, fastd_matype=0)
HSIdata["RSI"] = tl.RSI(HSIdata["Close"], timeperiod = 10)
HSIdata["MACD"], _, _ = tl.MACD(HSIdata["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
HSIdata["WILLR"] = tl.WILLR(HSIdata["High"], HSIdata["Low"], HSIdata["Close"], timeperiod=10)
HSIdata["ADOSC"] = tl.AD(HSIdata["High"], HSIdata["Low"], HSIdata["Close"], HSIdata["Volume"])
HSIdata["CCI"] = tl.CCI(HSIdata["High"], HSIdata["Low"], HSIdata["Close"], timeperiod=10)

#时间序列计算导致数据首尾出现新的NaN，需要进行遗失值补差
SMA = HSIdata["SMA"].values.reshape(-1,1)
WMA = HSIdata["WMA"].values.reshape(-1,1)
Momentum = HSIdata["Momentum"].values.reshape(-1,1)
K = HSIdata["K"].values.reshape(-1,1)
D = HSIdata["D"].values.reshape(-1,1)
RSI = HSIdata["RSI"].values.reshape(-1,1)
MACD = HSIdata["MACD"].values.reshape(-1,1)
WILLR = HSIdata["WILLR"].values.reshape(-1,1)
ADOSC = HSIdata["ADOSC"].values.reshape(-1,1)
CCI = HSIdata["CCI"].values.reshape(-1,1)

HSIdata["SMA"] = HSI_imputer.fit_transform(SMA)
HSIdata["WMA"] = HSI_imputer.fit_transform(WMA)
HSIdata["Momentum"] = HSI_imputer.fit_transform(Momentum)
HSIdata["K"] = HSI_imputer.fit_transform(K)
HSIdata["D"] = HSI_imputer.fit_transform(D)
HSIdata["RSI"] = HSI_imputer.fit_transform(RSI)
HSIdata["MACD"] = HSI_imputer.fit_transform(MACD)
HSIdata["WILLR"] = HSI_imputer.fit_transform(WILLR)
HSIdata["ADOSC"] = HSI_imputer.fit_transform(ADOSC)
HSIdata["CCI"] = HSI_imputer.fit_transform(CCI)

print("Data is successfully processed!")



from sklearn.model_selection import train_test_split
import torch
from torch import nn

HSIdata = np.array(HSIdata,type(float))
HSI_X = HSIdata[:,8:18]#此处选取的是计算所得的10个因子
HSI_Y = HSIdata[:,5]#此处选取的是Movement

X_trn, X_tst, Y_trn, Y_tst = train_test_split(HSI_X, HSI_Y, test_size=0.3)
X_trn, X_val, Y_trn, Y_val = train_test_split(X_trn, Y_trn, test_size=0.5)

from sklearn.preprocessing import Normalizer#为了降低数据复杂性，进行正则化
norm_X_trn = Normalizer().fit_transform(X_trn)
norm_X_tst = Normalizer().fit_transform(X_tst)
norm_X_val = Normalizer().fit_transform(X_val)


norm_X_trn = torch.from_numpy(norm_X_trn.astype(float)).type(torch.FloatTensor)
norm_X_tst = torch.from_numpy(norm_X_tst.astype(float)).type(torch.FloatTensor)
norm_X_val = torch.from_numpy(norm_X_val.astype(float)).type(torch.FloatTensor)
Y_trn = torch.from_numpy(Y_trn.astype(float)).type(torch.LongTensor)
Y_tst = torch.from_numpy(Y_tst.astype(float)).type(torch.LongTensor)
Y_val = torch.from_numpy(Y_val.astype(float)).type(torch.LongTensor)


import torch
from torch import nn
import torch.utils.data as Data

NW = nn.Sequential(
        nn.Linear(10, 256),
        nn.ReLU(True),
        nn.Linear(256, 256),
        nn.ReLU(True),
        nn.Linear(256, 2)
        )
Criterion = nn.CrossEntropyLoss()
Optimizer = torch.optim.SGD(NW.parameters(), lr=0.2)

Batch_Size = 32
torch_dataset = Data.TensorDataset(norm_X_trn, Y_trn)
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = Batch_Size,
    shuffle = False,
    num_workers = 2
    )

for epoch in range(1000):
    for step, (batch_x, batch_y) in enumerate(loader):
        batch_y_out = NW(batch_x).squeeze(-1)
        batch_loss = Criterion(batch_y_out, batch_y)
        if (step % 30 == 0):
            print("第{}次迭代的loss是:{}".format(step, batch_loss))
        Optimizer.zero_grad()
        batch_loss.backward()
        Optimizer.step()
    
        