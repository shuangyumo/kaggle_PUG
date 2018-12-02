import pandas as pd
from sklearn import preprocessing
import numpy as np
# 计算分差
## 注意matchDuration和numgroups每局对战都是一样的数据
def get_Rankdiff(df,group_target,cols):
    rela=lambda x:x-x.max()
    cols_rela=df[cols+[group_target]].groupby(group_target).transform(rela)
    cols=[col+'_diff' for col in cols]
    df[cols]=cols_rela
    return df

# 计算相对大小，用除法
def get_Relative(df,group_target,cols):
    rela=lambda x:x/x.max() if x.max()!=0 else x
    cols_rela=df[cols+[group_target]].groupby(group_target).transform(rela)
    cols=[col+'_rela'for col in cols]
    df[cols]=cols_rela
    return df










### groupid 取统计值   当使用这里的时候，就要注意下train['fea-min'] = train[[str(1+i) +'_total_fee' for i in range(4)]].min(axis = 1) 有没有这个必要看行特征
'''
columns :list
value:str
cname:str
'''
# 统计特征
def get_feat_stat_feat(train, base_feat, other_feat, stat_list):
    '''
    :param train:
    :param base_feat: list
    :param other_feat: list
    :param stat_list: list
    :return:
    '''
    result = pd.DataFrame()
    for stat in stat_list:
        agg_dict = {name: stat for name in other_feat}
        temp = train[base_feat + other_feat].groupby(base_feat)[other_feat].agg(agg_dict)
        temp.columns = [col + '_' + stat for col in temp.columns]
        result = pd.concat([result, temp], axis=1)
    return result

def merge_count(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    add.columns = columns + [cname]
    return add


def merge_nunique(df, columns, value, cname):
    add = pd.DataFrame(df.groupby(columns)[value].nunique()).reset_index()
    add.columns = columns + [cname]
    return add
# 划百分点
def merge_quantile(df, df_feature, fe, value, n, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].quantile(n)).reset_index()
    if not name:
        df_count.columns = fe + [value + "_%s_quantile" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    return df_count


#  独热编码
# one_hot编码，不常见的不编，只编常见的，然后不常见的选择性是否当成一类
def one_hot_encoder(train,column,n=100,nan_as_category=False):
    tmp = train[column].value_counts().to_frame()
    values = list(tmp[tmp[column]>n].index)
    train.loc[train[column].isin(values),column+'N'] = train.loc[train[column].isin(values),column]
    train =  pd.get_dummies(train, columns=[column+'N'], dummy_na=nan_as_category)
    return train

# label编码
def encode_count(df,column_name):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df[column_name].values))
    df[column_name] = lbl.transform(list(df[column_name].values))
    return df



# 获取某几列不同值的数量
def feature_num(df,cols,axis=1):
    return df[cols].nunique(axis=1)

# n_group Number each group from 0 to （the number of groups - 1.） 就是给每个group进行labelencode编号
# 要做肯定不是单独分一列，分一列就是label_encode，但是可以分多列的组合分
def get_feat_ngroup(train,base_feat):
    name = ('_').join(base_feat)+'_ngroup'

    train[name] = train.groupby(base_feat).ngroup()
    result = train[[name]]
    train.drop([name],axis=1,inplace=True)
    return result
