
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
def group_max(df,columns):
    '''

    :param df:
    :param columns: list,需要处理的列
    :return: new df
    '''
    agg=df.groupby('groupId')[columns].max()
    new_column=[]
    for column in columns:
        new_column.append('groupMax_'+column)
    agg.columns=new_column
    df=df.merge(agg,how='left',on='groupId')
    return df
def group_min(df,columns):
    '''

    :param df:
    :param columns: list,需要处理的列
    :return: new df
    '''
    agg=df.groupby('groupId')[columns].min()
    new_column=[]
    for column in columns:
        new_column.append('groupMin_'+column)
    agg.columns=new_column
    df=df.merge(agg,how='left',on='groupId')
    return df
def group_mean(df,columns):
    '''

    :param df:
    :param columns: list,需要处理的列
    :return: new df
    '''
    agg=df.groupby('groupId')[columns].mean()
    new_column=[]
    for column in columns:
        new_column.append('groupMean_'+column)
    agg.columns=new_column
    df=df.merge(agg,how='left',on='groupId')
    return df
if __name__ == '__main__':


    # train['totalDistance'] = 0.9 * train['walkDistance'] + 0.7 * train['swimDistance'] + train['rideDistance'] * 0.1
    # max_columns=['kills','damageDealt','walkDistance','killPlace']
    # min_columns=['killPlace']
    # max_columns=['killPlace']
    #
    # train=group_max(train,max_columns)
    # train=group_min(train,min_columns)
    # corr=train[['winPlacePerc','kills','damageDealt','walkDistance','killPlace',
    #       'group_kills', 'group_damageDealt', 'group_walkDistance',
    #        'group_killPlace']].corr()
    # print(corr)

    # 载入数据
    df = reload('../data/train_V2.csv', -1)
    invalid_match_ids = df[df['winPlacePerc'].isna()]['matchId'].values
    df = df[-df['matchId'].isin(invalid_match_ids)]
    print(df.shape)

    test_df = reload('../data/test_V2.csv', -1)
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
    max_columns = ['killPlace','damageDealt','walkDistance','rideDistance','assists','kills']
    df = group_min(df, max_columns)
    df = group_max(df, max_columns)
    df = group_mean(df, max_columns)
    df=df.iloc[0:9999]
    gc.collect()
    rankPoints = df['rankPoints']
    rankPoints[rankPoints.values == -1] = 0
    df['winPoints'] = rankPoints + df['winPoints']
    df['killPoints'] = df['killPoints'] + rankPoints
    df.drop(columns=['rankPoints'], inplace=True, axis=1)

    print(df.columns)
    ####处理matchtype
    ####只考虑solo,solo-fpp
    df.loc[(df['matchType'] == 'solo') | (df['matchType'] == 'solo-fpp') | (df['matchType'] == 'normal-solo-fpp')
           | (df['matchType'] == 'normal-solo'), 'matchType'] = 0
    # ####只考虑solo,
    # # df=df.loc[(df['matchType']=='solo')]
    # ####只考虑duo,
    df.loc[(df['matchType'] == 'duo') | (df['matchType'] == 'duo-fpp') | (df['matchType'] == 'normal-duo-fpp')
           | (df['matchType'] == 'normal-duo'), 'matchType'] = 1
    df.loc[(df['matchType'] == 'squad') | (df['matchType'] == 'squad-fpp') | (df['matchType'] == 'normal-squad-fpp')
           | (df['matchType'] == 'normal-squad'), 'matchType'] = 2
    df.loc[(df['matchType'] == 'crashfpp') | (df['matchType'] == 'crashtpp'), 'matchType'] = 3

    df.loc[(df['matchType'] == 'flarefpp') | (df['matchType'] == 'flaretpp'), 'matchType'] = 3

    print(df.shape)
    print(df.columns)


    ####把id,groupId,matchId去掉。
    df.drop(columns=['Id', 'groupId', 'matchId'], inplace=True, axis=1)
    # # 处理是否有车

    df['hasride'] = df['rideDistance']
    df['hasride'][df['hasride'] > 0] = 1

    ####加一个总距离
    df['totalDistance'] = df['rideDistance'] + df['walkDistance'] + df['swimDistance']
    # df.drop(columns=['killPlace'],inplace=True,axis=1)
    # df.drop(columns=['rideDistance','walkDistance','swimDistance'],inplace=True,axis=1)
    qcut_columns = ['damageDealt', 'longestKill', 'walkDistance', 'rideDistance', 'swimDistance', 'totalDistance',
                    'numGroups']
    cut_columns = ['matchDuration', 'maxPlace']
    df = max_corr_qcut(df, qcut_columns, 20, 1)
    df = max_corr_cut(df, cut_columns, 20, 1)
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
    train_target['winPlacePerc'] = target_pre
    train_target.to_csv("baseline_50.csv", index=False)

   ####感觉好奇怪，，我之前想的是因为排名是由团队最高排名那个决定的，为什么killPlace 取 最大值比最小值相关性
   # 高????           从killPlace 重要性感觉可以根据matchId 进行分组，排序，得到很多特征。
   #####10000行，各种信息都加上，，之前离散化处理后的也加上看看，发现有些离散化是没有原数据重要的。
# Fold
# 3
# macro - f1: 0.053252
# cv_result[0.05313, 0.0541, 0.05325]
