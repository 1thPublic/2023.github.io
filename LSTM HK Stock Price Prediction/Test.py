# -*- coding: utf-8 -*-
"""
Created on Thu May 26 23:36:02 2022

@author: windows
"""

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
HSIdata = pd.read_csv("K线导出_00981_日线数据.csv")

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

from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn

sel_col = ["SMA", "WMA", "Momentum", "K", "D", "RSI", "MACD", "WILLR", "ADOSC", "CCI"]
HSIdata = HSIdata[["Movement"]+ sel_col]
scaler = MinMaxScaler(feature_range=(-1, 1))

for col in sel_col:
    HSIdata[col] = scaler.fit_transform(HSIdata[col].values.reshape(-1,1))

HSIdata = HSIdata.astype(np.float32)

def create_seq_data(data_raw,seq):
    data_feat,data_target = [],[]
    for index in range(len(data_raw) - seq):
        # 构建特征集
        data_feat.append(data_raw[["SMA", "WMA", "Momentum", "K", "D", "RSI", "MACD", "WILLR", "ADOSC", "CCI"]][index: index + seq].values)
        # 构建target集
        data_target.append(data_raw["Movement"][index:index + seq])
    data_feat = np.array(data_feat)
    data_target = np.array(data_target)
    return data_feat,data_target


def train_test(data_feat,data_target,test_set_size,seq):
    train_size = data_feat.shape[0] - (test_set_size) 
    trainX = torch.from_numpy(data_feat[:train_size].reshape(-1,seq,10)).type(torch.FloatTensor)
    testX  = torch.from_numpy(data_feat[train_size:].reshape(-1,seq,10)).type(torch.FloatTensor)
    trainY = torch.from_numpy(data_target[:train_size].reshape(-1,seq,1)).type(torch.LongTensor)
    testY  = torch.from_numpy(data_target[train_size:].reshape(-1,seq,1)).type(torch.LongTensor)
    return trainX,trainY,testX,testY

n_time_steps = 10
test_set_size = int(np.round(0.2*HSIdata.shape[0]))
feat, target = create_seq_data(HSIdata, n_time_steps)
trainX, trainY, testX, testY = train_test(feat, target, test_set_size, n_time_steps)

print('x_train.shape = ',trainX.shape)
print('y_train.shape = ',trainY.shape)
print('x_test.shape = ',testX.shape)
print('y_test.shape = ',testY.shape)    

batch_size = 1442
num_epochs = 160
 
train = torch.utils.data.TensorDataset(trainX,trainY)
test = torch.utils.data.TensorDataset(testX,testY)
train_loader = torch.utils.data.DataLoader(dataset=train, 
                                           batch_size=batch_size, 
                                           shuffle=False)
 
test_loader = torch.utils.data.DataLoader(dataset=test, 
                                          batch_size=batch_size, 
                                          shuffle=False)


input_dim = 10
hidden_dim = 32
num_layers = 2 
output_dim = 2


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
 
        # Number of hidden layers
        self.num_layers = num_layers
 
        # Building LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
 
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
 
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
 
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
 
        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
 
        out = self.fc(out) 
        return out 


model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
print(model)

loss_fn = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

hist = np.zeros(num_epochs)
seq_dim = n_time_steps
for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    #model.hidden = model.init_hidden()
    
    # Forward pass
    y_train_pred = model(trainX)
 
    loss = loss_fn(y_train_pred[:,-1,:], trainY[:,-1,:].squeeze(-1))
    if t % 10 == 0 and t !=0:
        print("Epoch ", t, "CEL: ", loss.item())
    hist[t] = loss.item()
 
    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()
 
    # Backward pass
    loss.backward()
 
    # Update parameters
    optimiser.step()
    
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

Y_test_pred = model(testX)
Y_test_pred = torch.max(nn.functional.softmax(Y_test_pred[:,-1,:]), 1)[1]

confusion = confusion_matrix(testY[:,-1,:], Y_test_pred)
plt.imshow(confusion, cmap=plt.cm.Blues)
plt.xticks([0,1], ("Down", "Up"))
plt.yticks([0,1], ("Down", "Up"))
plt.colorbar()
plt.xlabel('Prediction')
plt.ylabel('True Value')
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[first_index][second_index])
plt.show()

acc = accuracy_score(testY[:,-1,:].squeeze(-1), Y_test_pred, normalize=True, sample_weight=None)
print("The prediction accuracy is", acc)
f1 = f1_score(testY[:,-1,:].squeeze(-1), Y_test_pred, average='binary')
print("The prediction f1-score is", f1)
if acc > 0.6 and f1 > 0.6:
    print("Good work man, we got it!")



    
