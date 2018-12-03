# Load libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import gc
import warnings

warnings.filterwarnings('ignore')

from utils import *
from xgb import *
from feature import *
from continues_to_discrete import *
def encode_onehot(df,column_name):
    feature_df=pd.get_dummies(df[column_name], prefix=column_name)
    all = pd.concat([df.drop([column_name], axis=1),feature_df], axis=1)
    return all

#载入数据
df = reload('../data/train_V2.csv', 10000)
invalid_match_ids = df[df['winPlacePerc'].isna()]['matchId'].values
df = df[-df['matchId'].isin(invalid_match_ids)]
print(df.shape)

test_df = reload('../data/test_V2.csv', 10000)
test_df['winPlacePerc'] = -1

print(test_df.shape)
df = pd.concat([df, test_df])
print(df.shape)
del test_df
gc.collect()
#
# # # 原始特征29
# # #
# print('1')
# # 处理winPoint和rankPoint，合并为一列  特征29-1=28
rankPoints = df['rankPoints']
rankPoints[rankPoints.values == -1] = 0
df['winPoints'] = rankPoints + df['winPoints']
df['killPoints']=df['killPoints']+rankPoints
df.drop(columns=['rankPoints'], inplace=True, axis=1)


####处理matchtype
df=encode_onehot(df,'matchType')


####把id,groupId,matchId去掉。
df.drop(columns = ['Id', 'groupId', 'matchId'], inplace = True, axis = 1)
# # 处理是否有车

df['hasride'] = df['rideDistance']
df['hasride'][df['hasride'] > 0] = 1

####加一个总距离
df['totalDistance']=df['rideDistance']+df['walkDistance']+df['swimDistance']
qcut_columns = ['damageDealt', 'longestKill', 'rideDistance','totalDistance',
                'swimDistance', 'walkDistance','numGroups']
cut_columns=['matchDuration']
df=max_corr_qcut(df,qcut_columns,20,1)
df=max_corr_cut(df,cut_columns,20,1)
print('7')
# 抽取y
train_data = df[df['winPlacePerc'] != -1]
test_data = df[df['winPlacePerc'] == -1]



train_target = train_data['winPlacePerc']
# 删除y
train_data.drop(columns=['winPlacePerc'], inplace=True, axis=1)
test_data.drop(columns=['winPlacePerc'], inplace=True, axis=1)

feature_names = train_data.columns
train_data = train_data.values

train_target_val = train_target.values
test_data = test_data.values


gc.collect()

print('训练开始')
parameters = {'nthread': -1,  # cpu 线程数 默认最大
              'objective': 'reg:linear',  # 多分类or 回归的问题    若要自定义就替换为custom_loss（不带引号）
              'learning_rate': .03,  # so called `eta` value 如同学习率
              'max_depth': 8,  # 构建树的深度，越大越容易过拟合
              'min_child_weight': 4,
              # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
              # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
              # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
              'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
              'subsample': 0.7,  # 随机采样训练样本
              'colsample_bytree': 0.7,  # 生成树时进行的列采样
              'n_estimators': 100000,  # 树的个数跟num_boost_round是一样的，所以可以设置无限大，靠early_stop
              'gamma': 0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
              'seed': 1000  # 随机种子
              # 'alpha':0, # L1 正则项参数
              # 'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
              # 'num_class':10, # 类别数，多分类与 multisoftmax 并用
              }
early_stopping_rounds = 100
num_folds = 3
eval_metric = mse
reg = get_XgbRegressor(train_data, train_target_val, test_data, feature_names, parameters,
                       early_stopping_rounds, num_folds, eval_metric, model_name='xgbbase1', stratified=False)
target_pre = reg.predict(test_data)
train_target['winPlacePerc']=target_pre
train_target.to_csv("baseline_50.csv",index=False)
####最终的结果，不是最终的，，，跑了一晚上的结果
# [4618]	validation_0-rmse:0.083366	validation_1-rmse:0.085676	validation_0-mse:0.060427	validation_1-mse:0.061422
# 0.060426883
# 0.061422072








