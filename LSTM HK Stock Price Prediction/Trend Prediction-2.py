# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 19:34:56 2022

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

HSIdata = np.array(HSIdata,type(float))
HSI_X = HSIdata[:,8:18]#此处选取的是计算所得的10个因子
HSI_Y = HSIdata[:,5]#此处选取的是Movement

X_trn, X_tst, Y_trn, Y_tst = train_test_split(HSI_X, HSI_Y, test_size=0.3)
X_trn, X_val, Y_trn, Y_val = train_test_split(X_trn, Y_trn, test_size=0.5)

X_trn = torch.from_numpy(X_trn.astype(float)).type(torch.FloatTensor)
X_tst = torch.from_numpy(X_tst.astype(float)).type(torch.FloatTensor)
X_val = torch.from_numpy(X_val.astype(float)).type(torch.FloatTensor)
Y_trn = torch.from_numpy(Y_trn.astype(float)).type(torch.LongTensor)
Y_tst = torch.from_numpy(Y_tst.astype(float)).type(torch.LongTensor)
Y_val = torch.from_numpy(Y_val.astype(float)).type(torch.LongTensor)

print("Data is successfully sampled!")

#先对模型进行初步的搭建
#已知输入层为10，输出层为1，暂时固定学习率为0.1
#由于分析连续问题，故采用ReLU为激活函数
#此模型采用了MSE来判拟合优度，通过Adam方法进行算法优化
#分别分析该模型在不同隐藏层神经元数量和epoch次数下的表现

class Net(nn.Module):
    def init_model(self, in_dim, n_hidden_1, n_hidden_2, out_dim, learn_rate):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.BatchNorm1d(in_dim,momentum=0.5),
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True),
            nn.BatchNorm1d(n_hidden_1,momentum=0.5),
            nn.Dropout(0.3),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.BatchNorm1d(n_hidden_2,momentum=0.5),
            nn.Dropout(0.3),
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

if __name__ == "__main__":
    HSI = Net()
    input_layer = 10
    output_layer = 2
    learn_rate = 0.1
    #首先假设两层的神经元个数相同，判别最优神经元个数和优化次数
    n_hidden = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    epoch = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    HSI_all_loss = []
    
    for i in n_hidden:
        HSI.init_model(in_dim = input_layer, n_hidden_1 = i, 
                       n_hidden_2 = i, out_dim = output_layer, learn_rate= 0.1)
        HSI_iter_loss = []
        HSI.train(x_trn = X_trn, y_trn = Y_trn, 
                  x_val = X_val, y_val = Y_val, epoch = 10000)
        HSI_all_loss.extend(HSI_iter_loss)
        
    best_hl_1 = n_hidden[np.argmin(HSI_all_loss) // 10]
    best_ep = epoch[np.argmin(HSI_all_loss) % 10]
    
    print("The selected number for hidden layer  c1 is", best_hl_1)
    print("The selected number for epoch is", best_ep)
    
    #确定最优第二隐藏层神经元个数
    n_hidden_2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    HSI_iter_loss = []
    
    for i in n_hidden:
        HSI.init_model(in_dim = input_layer, n_hidden_1 = best_hl_1, 
                       n_hidden_2 = i, out_dim = output_layer, learn_rate= 0.1)
        HSI.train(x_trn = X_trn, y_trn = Y_trn,
                  x_val = X_val, y_val = Y_val, epoch = best_ep)
        
    best_hl_2 = n_hidden[np.argmin(HSI_iter_loss) // 10]
    print("The selected number for hidden layer 2 is", best_hl_2)
    
    #通过最优参数模型，选取学习率
    learn_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
    HSI_iter_loss = []
    
    for k in learn_rate:
        HSI.init_model(in_dim = input_layer, n_hidden_1 = best_hl_1, 
                   n_hidden_2 = best_hl_2, out_dim = output_layer, learn_rate= k)
        HSI.train(x_trn = X_trn, y_trn = Y_trn,
                  x_val = X_val, y_val = Y_val, epoch = best_ep)
    
    best_lr = learn_rate[np.argmin(HSI_iter_loss)]
    print("The selected learn rate is", best_lr)
    
    #保存模型
    HSI.init_model(in_dim = input_layer, n_hidden_1 = best_hl_1, 
                   n_hidden_2 = best_hl_2, out_dim = output_layer, learn_rate= best_lr)
    HSI.train(x_trn = X_trn, y_trn = Y_trn,
              x_val = X_val, y_val = Y_val, epoch = best_ep)
    torch.save(HSI.model, "HSI.pt")
    def load_model(weights_name = "HSI.pt", epoch = best_ep, learn_rate = best_lr):
        HSI.model = torch.load(weights_name)
        HSI.criterion = nn.MSELoss()
        HSI.optimizer = torch.optim.Adam(HSI.model.parameters(),lr = learn_rate)
    
    #现在观察最佳模型在test_set上的表现
    load_model()
    Y_tst_out = HSI.model(X_tst)
    Y_tst_pred = torch.max(nn.functional.softmax(Y_tst_out), 1)[1]
    
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    
    confusion = confusion_matrix(Y_tst, Y_tst_pred)
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

    
    acc = accuracy_score(Y_tst, Y_tst_pred, normalize=True, sample_weight=None)
    print("The prediction accuracy is", acc)
    f1 = f1_score(Y_tst, Y_tst_pred, average='binary')
    print("The prediction f1-score is", f1)
    if acc > 0.6 and f1 > 0.6:
        print("Good work man, we got it!")
    
    # SSE.600000
    # n1 = n2 = 20
    # epb = 7000
    # lrb = 0.2
    