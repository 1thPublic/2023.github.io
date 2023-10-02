import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

train = pd.read_csv('D:/文件柜/研究/CTR with DeepFM/quick load/new_train.csv')
test = pd.read_csv('D:/文件柜/研究/CTR with DeepFM/quick load/new_test.csv')

pre_X = train[train['click'] == 0].sample(n=len(train[train['click'] == 1]), random_state=111)
pre_X = pd.concat([pre_X, train[train['click'] == 1]]).sample(frac=1)
pre_y = pre_X[['click']]
pre_X.drop(['click'], axis=1, inplace=True)
test.drop(['click'], axis=1, inplace=True)

for feat in pre_X.columns.tolist():
    lbe = LabelEncoder()
    pre_X[feat] = lbe.fit_transform(pre_X[feat])

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=pre_X[feat].max, embedding_dim=4, use_hash=True, dtype='string')
                          # since the input is string
                          for feat in pre_X.columns.tolist()]

linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )
    
pre_X_train, pre_X_test, pre_y_train, pre_y_test = train_test_split(pre_X, pre_y, test_size=0.20, stratify=pre_y, random_state=1)

# 4.Define Model,train,predict and evaluate
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy",
              metrics=['binary_crossentropy'], )

history = model.fit(pre_X_train, pre_y_train.values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
pred_ans = model.predict(pre_X_test, batch_size=256)
print("test LogLoss", round(log_loss(pre_y_test.values, pred_ans), 4))
print("test AUC", round(roc_auc_score(pre_y_test.values, pred_ans), 4))