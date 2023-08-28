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

HSIdata = pd.read_csv("K线导出_SPX_日线数据.csv")

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
MovPer = HSIdata["MovPer"].values.reshape(-1,1)
Volume = HSIdata["Volume"].values.reshape(-1,1)
Turnover = HSIdata["Turnover"].values.reshape(-1,1)

print("!")

HSI_imputer = SimpleImputer(missing_values=np.nan, strategy="median")

HSIdata["Open"] = HSI_imputer.fit_transform(Open)
HSIdata["Low"] = HSI_imputer.fit_transform(Low)
HSIdata["High"] = HSI_imputer.fit_transform(High)
HSIdata["Close"] = HSI_imputer.fit_transform(Close)
HSIdata["Movement"] = HSI_imputer.fit_transform(Movement)
HSIdata["MovPer"] = HSI_imputer.fit_transform(MovPer)
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

# #对10组技术指标进行分析
# print("SMA's Max is{0}, Min is {1}, Mean is {2}, Standard devision is{3}.".
#       format(np.max(SMA), np.min(SMA), np.mean(SMA), np.std(SMA))) 
# print("WMA's Max is{0}, Min is {1}, Mean is {2}, Standard devision is{3}.".
#       format(np.max(WMA), np.min(WMA), np.mean(WMA), np.std(WMA)))
# print("Momentum's Max is{0}, Min is {1}, Mean is {2}, Standard devision is{3}.".
#       format(np.max(Momentum), np.min(Momentum), np.mean(Momentum), np.std(Momentum)))
# print("K's Max is{0}, Min is {1}, Mean is {2}, Standard devision is{3}.".
#       format(np.max(K), np.min(K), np.mean(K), np.std(K)))
# print("D's Max is{0}, Min is {1}, Mean is {2}, Standard devision is{3}.".
#       format(np.max(D), np.min(D), np.mean(D), np.std(D)))
# print("RSI's Max is{0}, Min is {1}, Mean is {2}, Standard devision is{3}.".
#       format(np.max(RSI), np.min(RSI), np.mean(RSI), np.std(RSI)))
# print("MACD's Max is{0}, Min is {1}, Mean is {2}, Standard devision is{3}.".
#       format(np.max(MACD), np.min(MACD), np.mean(MACD), np.std(MACD)))
# print("WILLR's Max is{0}, Min is {1}, Mean is {2}, Standard devision is{3}.".
#       format(np.max(WILLR), np.min(WILLR), np.mean(WILLR), np.std(WILLR)))
# print("ADOSC's Max is{0}, Min is {1}, Mean is {2}, Standard devision is{3}.".
#       format(np.max(ADOSC), np.min(ADOSC), np.mean(ADOSC), np.std(ADOSC)))
# print("CCI's Max is{0}, Min is {1}, Mean is {2}, Standard devision is{3}.".
#       format(np.max(CCI), np.min(CCI), np.mean(CCI), np.std(CCI)))

# #观察10个指标之间的相关性，防止相关性过强的指标出现在输入变量中，影响结果
# import seaborn as sns
# import matplotlib.pyplot as plt
# import palettable

# HSI_indicators = HSIdata.iloc[:, 9:19]
# HSI_corr = HSI_indicators.corr(method='pearson')
# print(HSI_corr)
# plt.figure(figsize=(11, 9),dpi=100)
# sns.heatmap(data=HSI_corr,
#             vmax=0.3, 
#             cmap=palettable.cmocean.diverging.Curl_10.mpl_colors,
#             annot=True,
#             fmt=".2f",
#             annot_kws={'size':8,'weight':'normal', 'color':'#253D24'},
#             mask=np.triu(np.ones_like(HSI_corr,dtype=np.bool))#显示对脚线下面部分图
#            )

# #可以看到之前的指标选择造成很大的相关性，这是一段实验代码，用于找出相关性低的潜在指标


#截至到以上，indicators的计算已经完成，现在进行抽样方便之后的模型搭建
#ANN模型服从普通随机抽样，从总数据中抽取30%的数据为test_set
#其余数据50%为train_set（占总数据35%），50%为validation_set
from sklearn.model_selection import train_test_split
import torch
from torch import nn

HSIdata = np.array(HSIdata,type(float))
HSI_X = HSIdata[:,9:19]#此处选取的是计算所得的10个因子
HSI_Y = HSIdata[:,5]#此处选取的是Movement

HSI_X_2015 = HSI_X[0:94, :]
HSI_X_2016 = HSI_X[94:339, :]
HSI_X_2017 = HSI_X[339:585, :]
HSI_X_2018 = HSI_X[585:831, :]
HSI_X_2019 = HSI_X[831:1077, :]
HSI_X_2020 = HSI_X[1077:1325, :]
HSI_X_2021 = HSI_X[1325:1572, :]
HSI_X_2022 = HSI_X[1572:1642, :]

HSI_Y_2015 = HSI_Y[0:94]
HSI_Y_2016 = HSI_Y[94:339]
HSI_Y_2017 = HSI_Y[339:585]
HSI_Y_2018 = HSI_Y[585:831]
HSI_Y_2019 = HSI_Y[831:1077]
HSI_Y_2020 = HSI_Y[1077:1325]
HSI_Y_2021 = HSI_Y[1325:1572]
HSI_Y_2022 = HSI_Y[1572:1642]

X_trn_2015, X_tst_2015, Y_trn_2015, Y_tst_2015 = train_test_split(HSI_X_2015, HSI_Y_2015, test_size=0.3)
X_trn_2015, X_val_2015, Y_trn_2015, Y_val_2015 = train_test_split(X_trn_2015, Y_trn_2015, test_size=0.5)

X_trn_2016, X_tst_2016, Y_trn_2016, Y_tst_2016 = train_test_split(HSI_X_2016, HSI_Y_2016, test_size=0.3)
X_trn_2016, X_val_2016, Y_trn_2016, Y_val_2016 = train_test_split(X_trn_2016, Y_trn_2016, test_size=0.5)

X_trn_2017, X_tst_2017, Y_trn_2017, Y_tst_2017 = train_test_split(HSI_X_2017, HSI_Y_2017, test_size=0.3)
X_trn_2017, X_val_2017, Y_trn_2017, Y_val_2017 = train_test_split(X_trn_2017, Y_trn_2017, test_size=0.5)

X_trn_2018, X_tst_2018, Y_trn_2018, Y_tst_2018 = train_test_split(HSI_X_2018, HSI_Y_2018, test_size=0.3)
X_trn_2018, X_val_2018, Y_trn_2018, Y_val_2018 = train_test_split(X_trn_2018, Y_trn_2018, test_size=0.5)

X_trn_2019, X_tst_2019, Y_trn_2019, Y_tst_2019 = train_test_split(HSI_X_2019, HSI_Y_2019, test_size=0.3)
X_trn_2019, X_val_2019, Y_trn_2019, Y_val_2019 = train_test_split(X_trn_2019, Y_trn_2019, test_size=0.5)

X_trn_2020, X_tst_2020, Y_trn_2020, Y_tst_2020 = train_test_split(HSI_X_2020, HSI_Y_2020, test_size=0.3)
X_trn_2020, X_val_2020, Y_trn_2020, Y_val_2020 = train_test_split(X_trn_2020, Y_trn_2020, test_size=0.5)

X_trn_2021, X_tst_2021, Y_trn_2021, Y_tst_2021 = train_test_split(HSI_X_2021, HSI_Y_2021, test_size=0.3)
X_trn_2021, X_val_2021, Y_trn_2021, Y_val_2021 = train_test_split(X_trn_2021, Y_trn_2021, test_size=0.5)

X_trn_2022, X_tst_2022, Y_trn_2022, Y_tst_2022 = train_test_split(HSI_X_2022, HSI_Y_2022, test_size=0.3)
X_trn_2022, X_val_2022, Y_trn_2022, Y_val_2022 = train_test_split(X_trn_2022, Y_trn_2022, test_size=0.5)

# print("In 2015, Close's mean and median for trn, val, tst are", 
#       np.mean(Y_trn_2015), np.median(Y_trn_2015), 
#       np.mean(Y_val_2015), np.median(Y_val_2015), 
#       np.mean(Y_tst_2015), np.median(Y_tst_2015))
# print("In 2016, Close's mean and median for trn, val, tst are", 
#       np.mean(Y_trn_2016), np.median(Y_trn_2016), 
#       np.mean(Y_val_2016), np.median(Y_val_2016), 
#       np.mean(Y_tst_2016), np.median(Y_tst_2016))
# print("In 2017, Close's mean and median for trn, val, tst are", 
#       np.mean(Y_trn_2017), np.median(Y_trn_2017), 
#       np.mean(Y_val_2017), np.median(Y_val_2017), 
#       np.mean(Y_tst_2017), np.median(Y_tst_2017))
# print("In 2018, Close's mean and median for trn, val, tst are", 
#       np.mean(Y_trn_2018), np.median(Y_trn_2018), 
#       np.mean(Y_val_2018), np.median(Y_val_2018), 
#       np.mean(Y_tst_2018), np.median(Y_tst_2018))
# print("In 2019, Close's mean and median for trn, val, tst are", 
#       np.mean(Y_trn_2019), np.median(Y_trn_2019), 
#       np.mean(Y_val_2019), np.median(Y_val_2019), 
#       np.mean(Y_tst_2019), np.median(Y_tst_2019))
# print("In 2020, Close's mean and median for trn, val, tst are", 
#       np.mean(Y_trn_2020), np.median(Y_trn_2020), 
#       np.mean(Y_val_2020), np.median(Y_val_2020), 
#       np.mean(Y_tst_2020), np.median(Y_tst_2020))
# print("In 2021, Close's mean and median for trn, val, tst are", 
#       np.mean(Y_trn_2021), np.median(Y_trn_2021), 
#       np.mean(Y_val_2021), np.median(Y_val_2021), 
#       np.mean(Y_tst_2021), np.median(Y_tst_2021))
# print("In 2022, Close's mean and median for trn, val, tst are", 
#       np.mean(Y_trn_2022), np.median(Y_trn_2022), 
#       np.mean(Y_val_2022), np.median(Y_val_2022), 
#       np.mean(Y_tst_2022), np.median(Y_tst_2022))

X_trn = np.r_[X_trn_2015, X_trn_2016, X_trn_2016, X_trn_2017, X_trn_2018, 
              X_trn_2019, X_trn_2020, X_trn_2021, X_trn_2022]
X_val = np.r_[X_val_2015, X_val_2016, X_val_2016, X_val_2017, X_val_2018, 
              X_val_2019, X_val_2020, X_val_2021, X_val_2022]
X_tst = np.r_[X_tst_2015, X_tst_2016, X_tst_2016, X_tst_2017, X_tst_2018, 
              X_tst_2019, X_tst_2020, X_tst_2021, X_tst_2022]
Y_trn = np.r_[Y_trn_2015, Y_trn_2016, Y_trn_2016, Y_trn_2017, Y_trn_2018, 
              Y_trn_2019, Y_trn_2020, Y_trn_2021, Y_trn_2022]
Y_val = np.r_[Y_val_2015, Y_val_2016, Y_val_2016, Y_val_2017, Y_val_2018, 
              Y_val_2019, Y_val_2020, Y_val_2021, Y_val_2022]
Y_tst = np.r_[Y_tst_2015, Y_tst_2016, Y_tst_2016, Y_tst_2017, Y_tst_2018, 
              Y_tst_2019, Y_tst_2020, Y_tst_2021, Y_tst_2022]

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

#先对模型进行初步的搭建
#已知输入层为10，输出层为1，暂时固定学习率为0.1
#由于分析连续问题，故采用ReLU为激活函数
#此模型采用了MSE来判拟合优度，通过Adam方法进行算法优化
#分别分析该模型在不同隐藏层神经元数量和epoch次数下的表现

class Net(nn.Module):
    def init_model(self, in_dim, n_hidden_1, n_hidden_2, out_dim, learn_rate):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Linear(n_hidden_2, out_dim)
             )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learn_rate)
    
        for m in self.modules():# 迭代循环初始化参数
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu'),
        
    def train(self, x, y, epoch):
        self.epoch = epoch
        for i in range(epoch):
            # forward
            y_out = self.model(x).squeeze(-1)
            # calc loss
            self.loss = self.criterion(y_out, y)
            if (i % 30 == 0):
                print("第{}次迭代的loss是:{}".format(i, self.loss))
            # zero grad
            self.optimizer.zero_grad()
            # backward
            self.loss.backward()
            # adjust weights
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
        for j in epoch:
            HSI.train(x = norm_X_trn, y = Y_trn, epoch = j)
            Y_val_pred = HSI.model(norm_X_val)
            HSI_val_loss = HSI.criterion(Y_val_pred, Y_val)
            HSI_iter_loss.append(HSI_val_loss.item())
        HSI_all_loss.extend(HSI_iter_loss)
        
    best_hl_1 = n_hidden[np.argmin(HSI_all_loss) // 10]
    best_ep = epoch[np.argmin(HSI_all_loss) % 10]
    
    print("The selected number for hidden layer 1 is", best_hl_1)
    print("The selected number for epoch is", best_ep)
    
    #确定最优第二隐藏层神经元个数
    n_hidden_2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    HSI_iter_loss = []
    
    for i in n_hidden:
        HSI.init_model(in_dim = input_layer, n_hidden_1 = best_hl_1, 
                       n_hidden_2 = i, out_dim = output_layer, learn_rate= 0.1)
        HSI.train(x = norm_X_trn, y = Y_trn, epoch = best_ep)
        Y_val_pred = HSI.model(norm_X_val)
        HSI_val_loss = HSI.criterion(Y_val_pred, Y_val)
        HSI_iter_loss.append(HSI_val_loss.item())
        
    best_hl_2 = n_hidden_2[np.argmin(HSI_iter_loss)]
    print("The selected number for hidden layer 2 is", best_hl_2)
    
    #通过最优参数模型，选取学习率
    learn_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    HSI_iter_loss = []
    for k in learn_rate:
        HSI.init_model(in_dim = input_layer, n_hidden_1 = best_hl_1, 
                   n_hidden_2 = best_hl_2, out_dim = output_layer, learn_rate= k)
        HSI.train(x = norm_X_trn, y = Y_trn, epoch = best_ep)
        Y_val_pred = HSI.model(norm_X_val)
        HSI_val_loss = HSI.criterion(Y_val_pred, Y_val)
        HSI_iter_loss.append(HSI_val_loss.item())
    
    best_lr = learn_rate[np.argmin(HSI_iter_loss)]
    print("The selected learn rate is", best_lr)
    
    #保存模型
    HSI.init_model(in_dim = input_layer, n_hidden_1 = best_hl_1, 
                   n_hidden_2 = best_hl_2, out_dim = output_layer, learn_rate= best_lr)
    HSI.train(norm_X_trn, Y_trn, best_ep)
    torch.save(HSI.model, "HSI.pt")
    def load_model(weights_name = "HSI.pt", epoch = best_ep, learn_rate = best_lr):
        HSI.model = torch.load(weights_name)
        HSI.criterion = nn.MSELoss()
        HSI.optimizer = torch.optim.Adam(HSI.model.parameters(),lr = learn_rate)
    
    #现在观察最佳模型在test_set上的表现
    load_model()
    Y_tst_out = HSI.model(norm_X_tst)
    Y_tst_pred = torch.max(nn.functional.softmax(Y_tst_out), 1)[1]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_tst, Y_tst_pred)
    
    print(cm)

