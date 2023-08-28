# -*- coding: utf-8 -*-
"""
Created on Mon May 23 18:53:15 2022

@author: windows
"""
#设置工作路径
import os

os.chdir("D:\文件柜\研究\港股互联网\数据处理")

# 这是用来做股指、股票数据处理的代码，目标效果如论文所示
# 首先将三组数据导入，按照时间序列对数据进行处理，使用循环语句算出所需9组指标
# 其次，将数据打散为独立，从每年的数据中抽取占样本总量20%的天数作为训练集和检测集
# 读取10年间HSI, HSIII指数以及腾讯和中芯国际的股价数据
# 数据源: Choice金融终端
import pandas as pd
HSIdata = pd.read_csv("K线导出_HSIII_日线数据.csv")

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
HSI_Y = HSIdata[:,4]#此处选取的是Movement


from sklearn.model_selection import train_test_split
import torch
from torch import nn
torch.set_default_tensor_type(torch.DoubleTensor)

HSIdata = np.array(HSIdata,type(float))
HSI_X = HSIdata[:,8:18]#此处选取的是计算所得的10个因子
HSI_Y = HSIdata[:,5]#此处选取的是Movement

X_tst, X_trn = HSI_X[0:int(1642*0.3),:], HSI_X[int(1642*0.3):,:]
Y_tst, Y_trn = HSI_Y[0:int(1642*0.3)], HSI_Y[int(1642*0.3):]
X_trn, X_val = X_trn[0:int(1642*0.7*0.5),:], X_trn[int(1642*0.7*0.5):,:]
Y_trn, Y_val = Y_trn[0:int(1642*0.7*0.5)], Y_trn[int(1642*0.7*0.5):]

n_time_steps = 3

X_trn = [torch.tensor(X_trn[i:i+n_time_steps,].astype(float)).view(1, n_time_steps, 10) 
         for i in range(len(X_trn)-n_time_steps)]
X_tst = [torch.tensor(X_tst[i:i+n_time_steps,].astype(float)).view(1, n_time_steps, 10) 
         for i in range(len(X_tst)-n_time_steps)]
X_val = [torch.tensor(X_val[i:i+n_time_steps,].astype(float)).view(1, n_time_steps, 10) 
         for i in range(len(X_val)-n_time_steps)]

Y_trn = [torch.tensor([Y_trn[i+n_time_steps - 1]]).type(torch.LongTensor)  
         for i in range(len(Y_trn)-n_time_steps)]
Y_tst = [torch.tensor([Y_tst[i+n_time_steps - 1]]).type(torch.LongTensor)  
         for i in range(len(Y_tst)-n_time_steps)]
Y_val = [torch.tensor([Y_val[i+n_time_steps - 1]]).type(torch.LongTensor) 
         for i in range(len(Y_val)-n_time_steps)]

X_tst = torch.tensor( [item.cpu().detach().numpy() for item in X_tst] )
X_val = torch.tensor( [item.cpu().detach().numpy() for item in X_val] )
Y_tst = torch.tensor( [item.cpu().detach().numpy() for item in Y_tst] )
Y_val = torch.tensor( [item.cpu().detach().numpy() for item in Y_val] )

X_tst = X_tst.view(489,3,10)
X_val = X_val.view(573,3,10)

import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class LayerNormalization(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNormalization, self).__init__()
        
        self.eps = eps
        self.hidden_size = hidden_size
        self.a2 = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        
    def forward(self, z):
        mu = torch.mean(z, dim=1)
        sigma = torch.std(z, dim=1)

        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a2 + self.b2
        return ln_out
    
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMCell, self).__init__()

        self.hidden_size = hidden_size
        # lstm weights
        self.weight_fh = nn.Linear(hidden_size, hidden_size)
        self.weight_ih = nn.Linear(hidden_size, hidden_size)
        self.weight_ch = nn.Linear(hidden_size, hidden_size)
        self.weight_oh = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(input_size, hidden_size)
        self.weight_ix = nn.Linear(input_size, hidden_size)
        self.weight_cx = nn.Linear(input_size, hidden_size)
        self.weight_ox = nn.Linear(input_size, hidden_size)
        # decoder
        self.decoder = nn.Linear(hidden_size, output_size)
        # layer normalization
        self.lnx = LayerNormalization(hidden_size)
        self.lnh = LayerNormalization(hidden_size)
        self.lnc = LayerNormalization(hidden_size)

    def forward(self, inp, h_0, c_0):
        # forget gate
        f_g = F.sigmoid(self.lnx(self.weight_fx(inp)) + self.lnh(self.weight_fh(h_0)))
        # input gate
        i_g = F.sigmoid(self.lnx(self.weight_ix(inp)) + self.lnh(self.weight_ih(h_0)))
        # intermediate cell state
        c_tilda = F.tanh(self.lnx(self.weight_cx(inp)) + self.lnh(self.weight_ch(h_0)))
        # current cell state
        cx = f_g * c_0 + i_g * c_tilda
        # output gate
        o_g = F.sigmoid(self.lnx(self.weight_ox(inp)) + self.lnh(self.weight_oh(h_0)))
        # hidden state
        hx = o_g * F.tanh(self.lnc(cx))
        hx1 = torch.Tensor(hx.tolist()[0][-1])

        out = self.decoder(hx1.view(1,-1))

        return out, hx, cx

    def init_hidden(self):
        h_0 = Variable(torch.zeros(1, self.hidden_size))
        c_0 = Variable(torch.zeros(1, self.hidden_size))
        return h_0, c_0

class LSTM(nn.Module):
    def init_model(self, input_size, hidden_size, output_size, learn_rate):
        self.RNN = LSTMCell(input_size, hidden_size, output_size)
        self.LM1 = LayerNormalization(hidden_size)
        self.LM2 = LayerNormalization(input_size)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.RNN.parameters(), lr=learn_rate)
    
    def forward(self, inp):
        hx, cx = self.RNN.init_hidden()
        for ipt in inp:
            ipt = self.LM2(ipt)
            hx = self.LM1(hx)
            cx = self.LM1(cx)
            y_out , hx, cx = self.RNN.forward(ipt, hx, cx)
        return y_out , hx, cx
        
    def train(self, x_trn, y_trn, x_val, y_val, epoch):

        for j in range(epoch):
            losses = []
            y_out , hx, cx = self.forward(x_trn[j])
            self.loss = self.criterion(y_out.detach(), y_trn[j])
            if (j+1) % 10 == 0:
                losses.append(self.loss.tolist())
            self.loss.requires_grad_(True)
            self.optimizer.zero_grad()
            self.loss.backward(retain_graph=True)
            self.optimizer.step()
            if (j+1) % 30 == 0:
                for k, inp in enumerate(x_val[j]):
                    y_val_out, _ = self.model(x_val[i], h_state)
                    y_val_out = self.fc(y_val_out[:, -1, :])
                    val_loss = self.criterion(y_val_out, y_val[i])
                    print("第{}次迭代的tst_loss是:{}".format(i+1, self.loss))
                    print("第{}次迭代的val_loss是:{}".format(i+1, val_loss))
                if i-1 == len(X_trn):
                    
            