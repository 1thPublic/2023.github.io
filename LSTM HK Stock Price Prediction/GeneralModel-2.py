# -*- coding: utf-8 -*-
"""
Created on Tue May 24 23:37:53 2022

@author: windows
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:57:25 2022

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
HSIdata = pd.read_csv("K线导出_HSI_日线数据.csv")

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

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, learn_rate):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True)
        for p in self.model.parameters():
          nn.init.normal_(p, mean=0.0, std=0.001)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.BN1 = nn.BatchNorm1d(input_size,momentum=0.5)
        self.BN2 = nn.BatchNorm1d(hidden_size,momentum=0.5)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate)
    
    def train(self, x_trn, y_trn, x_val, y_val, epoch):
        from sklearn.metrics import accuracy_score
        k = 0
        for j in range(int(len(x_trn)/30)):
            h_state = None
            for step in range(30):
                x_trn[k][0] = self.BN1(x_trn[k][0])
                y_out , h_state = self.model(x_trn[k], h_state)
                y_out[0] = self.BN2(y_out[0])
                y_out = self.fc(y_out[:, -1, :])
                self.loss = self.criterion(y_out.detach(), y_trn[k])
                self.loss.requires_grad_(True)
                self.optimizer.zero_grad()
                self.loss.backward(retain_graph=True)
                self.optimizer.step()
                k = k + 1
            
            h_state = None
            y_val_out, _ = self.model(x_val, h_state)
            y_val_out = self.fc(y_val_out[:, -1, :])
            y_pred = torch.max(nn.functional.softmax(y_val_out), 1)[1]
            acc = accuracy_score(y_val, y_pred, normalize=True, sample_weight=None)
            print("Epoch: ", epoch, "| Mini Batch: ", j+1, "| Val Accuracy: ", acc)
            RNN_iter_acc.append(acc)
                
if __name__ == "__main__":
    input_layer = 10
    output_layer = 2
    learn_rate = 0.1
    #首先假设两层的神经元个数相同，判别最优神经元个数和优化次数
    n_hidden = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    n_epoch = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    hidden_layer = 2
    RNN_all_acc = []
    
    for i in n_hidden:
        RNN_iter_acc = []
        RNN = LSTM(input_size = input_layer, 
                       hidden_size = i, 
                       num_layers = hidden_layer, 
                       num_classes = output_layer, 
                       learn_rate= 0.1)
        for j in range(2000):
            RNN.train(x_trn = X_trn, y_trn = Y_trn, 
                      x_val = X_val, y_val = Y_val,
                      epoch = j)
        RNN_all_acc.extend(RNN_iter_acc)   
        
    best_hl = n_hidden[np.argmin(RNN_all_acc) // 10]
    best_ep = n_epoch[np.argmin(RNN_all_acc) % 10]
    
    print("The selected number for hidden layers is", best_hl)
    print("The selected number for epoch is", best_ep)
    
    RNN_iter_acc = []
    for k in learn_rate:
        RNN.init_model(input_size = input_layer, 
                       hidden_size = best_hl, 
                       num_layers = hidden_layer, 
                       num_classes = output_layer,
                       learn_rate= k)
        RNN.train(x_trn = X_trn, y_trn = Y_trn,
                  x_val = X_val, y_val = Y_val, 
                  epoch = best_ep)
    
    best_lr = learn_rate[np.argmin(RNN_iter_acc)]
    print("The selected learn rate is", best_lr)