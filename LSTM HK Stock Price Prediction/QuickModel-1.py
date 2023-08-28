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

import torch
from torch import nn
class Net(nn.Module):
    def init_model(self, in_dim, n_hidden_1, n_hidden_2, out_dim, learn_rate):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.BatchNorm1d(in_dim,momentum=0.5),
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1,momentum=0.5),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2,momentum=0.5),
            nn.ReLU(True),
            nn.Linear(n_hidden_2, out_dim)
             )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learn_rate)
        
    def train(self, x_trn, y_trn, x_val, y_val, epoch):
        self.epoch = epoch
        for i in range(epoch):
            y_out = self.model(x_trn).squeeze(-1)
            self.loss = self.criterion(y_out, y_trn)
            y_val_out = self.model(x_val).squeeze(-1)
            val_loss = self.criterion(y_val_out, y_val)
            if ((i+1) % 30 == 0):
                print("第{}次迭代的tst_loss是:{}".format(i+1, self.loss))
                print("第{}次迭代的val_loss是:{}".format(i+1, val_loss))
            if ((i+1) % 1000 == 0):
                HSI_iter_loss.append(val_loss.item())
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            
    def calc_loss(self, y_pred, y_val):#此处用于通过val评估模型训练的表现
        y_pred = torch.from_numpy(y_pred).type(torch.FloatTensor)
        HSI_val_loss = self.criterion(y_pred, y_val)
        return HSI_val_loss
    
        