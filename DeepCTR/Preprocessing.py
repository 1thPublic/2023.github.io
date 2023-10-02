import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os # see all the files in the list
for dirname, _, filenames in os.walk('D:/文件柜/研究/CTR with DeepFM/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from matplotlib import pyplot as plt
import seaborn as sns

chunksize = 10 ** 6
num_of_chunk = 0
train = pd.DataFrame()
    
for chunk in pd.read_csv('D:/文件柜/研究/CTR with DeepFM/input/train.csv', chunksize=chunksize):
    num_of_chunk += 1
    train = pd.concat([train, chunk.sample(frac=.05, replace=False, random_state=123)], axis=0)
    print('Processing Chunk No. ' + str(num_of_chunk))     
    
train.reset_index(inplace=True)

train_len = len(train) # save the length for for future use

df = pd.concat([train, pd.read_csv('D:/文件柜/研究/CTR with DeepFM/input/test.csv')]).drop(['index', 'id'], axis=1)

def get_date(hour): # generate date
    y = '20'+str(hour)[:2]
    m = str(hour)[2:4]
    d = str(hour)[4:6]
    return y+'-'+m+'-'+d

df['weekday'] = pd.to_datetime(df.hour.apply(get_date)).dt.dayofweek.astype(str) # transfer date into weekday

def tran_hour(x): # generate hour section
    x = x % 100
    while x in [23,0]:
        return '23-01'
    while x in [1,2]:
        return '01-03'
    while x in [3,4]:
        return '03-05'
    while x in [5,6]:
        return '05-07'
    while x in [7,8]:
        return '07-09'
    while x in [9,10]:
        return '09-11'
    while x in [11,12]:
        return '11-13'
    while x in [13,14]:
        return '13-15'
    while x in [15,16]:
        return '15-17'
    while x in [17,18]:
        return '17-19'
    while x in [19,20]:
        return '19-21'
    while x in [21,22]:
        return '21-23'

df['hour'] = df.hour.apply(tran_hour) # transfer hour into hour section

df.info() # check the categories
len_of_feature_count = []
for i in df.columns[2:23].tolist(): # check the range length for variables
    print(i, ':', len(df[i].astype(str).value_counts()))
    len_of_feature_count.append(len(df[i].astype(str).value_counts()))

need_tran_feature = df.columns[2:4].tolist() + df.columns[13:23].tolist() # set up a list to store the float type variables

for i in need_tran_feature: # convert them into category
    df[i] = df[i].astype(str)

obj_features = []

for i in range(len(len_of_feature_count)):
    if len_of_feature_count[i] > 10:
        obj_features.append(df.columns[2:23].tolist()[i])
obj_features

df_describe = df.describe()
df_describe

def obj_clean(X):
    # define a function to reduce the range of values of features

    def get_click_rate(x): # function to get click rate
        temp = train[train[X.columns[0]] == x]
        res = round((temp.click.sum() / temp.click.count()),3)
        return res

    def get_type(V, str): # function to generate type determination rule for features
        very_high = df_describe.loc['mean','click'] + 0.04
        higher = df_describe.loc['mean','click'] + 0.02
        lower = df_describe.loc['mean','click'] - 0.02
        very_low = df_describe.loc['mean','click'] - 0.04

        vh_type = V[V[str] > very_high].index.tolist()
        hr_type = V[(V[str] > higher) & (V[str] < very_high)].index.tolist()
        vl_type = V[V[str] < very_low].index.tolist()
        lr_type = V[(V[str] < lower) & (V[str] > very_low)].index.tolist()

        return vh_type, hr_type, vl_type, lr_type

    def clean_function(x): # function to get the row type
        while x in type_[0]:
            return 'very_high'
        while x in type_[1]:
            return 'higher'
        while x in type_[2]:
            return 'very_low'
        while x in type_[3]:
            return 'lower'
        return 'mid'
        
    print('Run: ', X.columns[0])
    fq = X[X.columns[0]].value_counts() # save the frequency in a list
    if len(fq) > 1000: # to save time complexity, we focus on the rows within 1000 frequency
        fq = fq[:1000]

    fq = pd.DataFrame(fq) # convert frequency into a new dataframe
    fq['new_column'] = fq.index    

    fq['click_rate'] = fq.new_column.apply(get_click_rate) # apply function to each row, get click rates

    type_ = get_type(fq, 'click_rate')  # generate type determination rule for each feature

    return X[X.columns[0]].apply(clean_function) # apply function to each row, get the types

# generate a new dataframe to store the types
new = {}
for i in obj_features:    
    new[i] = obj_clean(df[[i]])

new = pd.DataFrame(new)
new

old_list = [item for item in df.columns.tolist() if item not in obj_features]
old = df[old_list]
old

new_df = pd.concat([old, new], axis=1)
new_df

for i in df.columns:
    sns.countplot(x = i, hue = "click", data = new_df)
    plt.show()

new_df.drop(['device_id', 'C14', 'C17', 'C19', 'C20', 'C21'], axis=1, inplace=True)

# split train and test sets again
train = new_df[:train_len]
test = new_df[train_len:]

train.to_csv('D:/文件柜/研究/CTR with DeepFM/quick load/new_train.csv', index=False)
test.to_csv('D:/文件柜/研究/CTR with DeepFM/quick load/new_test.csv', index=False)