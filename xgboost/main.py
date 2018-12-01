# Load libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import gc
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
from utils import *
from xgb import *
from feature import *




#载入数据
df = reload('../input/train_V2.csv', - 1)
# 去除为空的数据
invalid_match_ids = df[df['winPlacePerc'].isna()]['matchId'].values
df = df[-df['matchId'].isin(invalid_match_ids)]
print(df.shape)

test_df = reload('../input/test_V2.csv', -1)
test_df['winPlacePerc'] = -1

print(test_df.shape)
df = pd.concat([df, test_df])
print(df.shape)
del test_df
gc.collect()

# # 原始特征29
# #
print('1')
# 处理winPoint和rankPoint，合并为一列  特征29-1=28
rankPoints = df['rankPoints']
rankPoints[rankPoints.values == -1] = 0
df['winPoints'] = rankPoints + df['winPoints']
df['killPoints']=df['killPoints']+rankPoints
df.drop(columns=['rankPoints'], inplace=True, axis=1)





print('2')
# 爆头率，击杀率，短时间击杀率 28+3=31
df['headshotRate'] = df.headshotKills / df.kills
df['killRate'] = df.DBNOs / (df.kills + df.DBNOs)
df['streakRate'] = df.killStreaks / df.kills

print('3')
# matchId 就不要算相对值，给出一场比赛有多少人  31-1+1=31
df=encode_count(df,'matchId')
new_t=pd.DataFrame(df.matchId.value_counts().values,columns=['matchvalue'])
new_t['matchId']=df.matchId.value_counts().index
df=pd.merge(df,new_t,on='matchId')
df.drop(columns=['matchId'], inplace=True, axis=1)



print('4')
# # 玩家数量 1
new_df = merge_count(df, ['groupId'], 'assists', 'player_num')

# 把类别行的去掉  1+（31-5）*6=156
cols = list(set(df.columns.values) - set(['Id', 'groupId', 'winPlacePerc', 'matchType','matchvalue']))
for col in cols:
    new_df = pd.concat([new_df, merge_median(df, ['groupId'], col, col + 'median')],axis=1)
for col in cols:
    new_df = pd.concat([new_df, merge_mean(df, ['groupId'], col, col + 'mean')],axis=1)
for col in cols:
    new_df = pd.concat([new_df, merge_sum(df, ['groupId'], col, col + 'sum')],axis=1)
for col in cols:
    new_df = pd.concat([new_df, merge_max(df, ['groupId'], col, col + 'max')],axis=1)
for col in cols:
    new_df = pd.concat([new_df, merge_min(df, ['groupId'], col, col + 'min')],axis=1)
for col in cols:
    new_df = pd.concat([new_df, merge_std(df, ['groupId'], col, col + 'std')],axis=1)
print('5')
#156+2=158
new_df = pd.concat([new_df, merge_mean(df, ['groupId'], 'winPlacePerc', 'winPlacePerc')],axis=1)
new_df = pd.concat([new_df, merge_mean(df, ['groupId'], 'matchvalue', 'matchvalue')],axis=1)

# 处理是否有车
# 158+1=159
new_df['hasride'] = new_df['rideDistancemin']
new_df['hasride'][new_df['hasride'] > 0] = 1



del df
gc.collect()
print('6')
#删除object数据
#159-1=158
new_df.drop(columns=['groupId'], inplace=True, axis=1)
new_df.to_csv('f1.csv', index=False)

print('7')
# 抽取y
train_data = new_df[new_df['winPlacePerc'] != -1]
test_data = new_df[new_df['winPlacePerc'] == -1]



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
              'max_depth': 6,  # 构建树的深度，越大越容易过拟合
              'min_child_weight': 4,
              # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
              # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
              # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
              'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
              'subsample': 0.7,  # 随机采样训练样本
              'colsample_bytree': 0.7,  # 生成树时进行的列采样
              'n_estimators': 200,  # 树的个数
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





# target_pre = reg.predict(test_data)
# test_df = reload('../input/test_V2.csv', -1)
# test_df = test_df[['Id', 'groupId']]
# test_data = pd.DataFrame(test_data, columns=feature_names)
# test_data_val = test_data['groupId']
# test_data_val['winPlacePerc'] = target_pre
#
# result = pd.merge(test_df, test_data_val, on='groupId')
# result.to_csv("baseline_f1.csv", index=False)










