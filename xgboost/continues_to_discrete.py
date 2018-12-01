import pandas as pd
import numpy as np


####等频切分成K份
####这里会出问题，因为train['damageDealt].quantile(0.1 0.2 都是0，
def continues_qcut(df,columns,k=9):
    data=df.copy()
    for column in columns:
        num=k
        for i in range(k):
            if data[column].quantile(i/k)==data[column].quantile((i+1)/k):
                num=num-1
        columnName=column+'QCategory'
        data[columnName] = pd.qcut(data[column],k,labels=[i for i in range(num)],duplicates='drop')
        print(data[columnName])
        data[columnName] =data[columnName].to_frame()
        # print(data[columnName])
        # corr = data[['winPlacePerc', columnName]].corr()
        # print(corr)
    return data
def continues_cut(df,columns,k=9):
    data = df.copy()
    for column in columns:
        columnName = column + 'Category'
        data[columnName] = pd.cut(data[column], k, labels=[i for i in range(k)], duplicates='drop')
        data[columnName] = data[columnName].to_frame()
        # print(data[columnName])
        # corr = data[['winPlacePerc', columnName]].corr()
        # print(corr)
    return data
####kmax 是K值的上界,columns是传入的list列名
def max_corr_k_qcut(df,columns,kmax):
    k_values = {}
    corr_values={}
    for column in columns:
        k_values[column]=0
        corr_values[column]=0
    for k in range(1,kmax+1):
        data=continues_qcut(df,columns,k)
        for column in columns:
            corr=data[[column+'QCategory','winPlacePerc']].corr().iloc[0,1]
            print(corr)
            if corr_values[column]**2<corr**2:
                corr_values[column]=corr
                k_values[column]=k
    return k_values
def max_corr_k_cut(df,columns,kmax):
    k_values = {}
    corr_values={}
    for column in columns:
        k_values[column]=0
        corr_values[column]=0
    for k in range(1,kmax+1):
        data=continues_cut(df,columns,k)
        for column in columns:
            corr=data[[column+'QCategory','winPlacePerc']].corr().iloc[0,1]
            print(corr)
            if corr_values[column]**2<corr**2:
                corr_values[column]=corr
                k_values[column]=k
    return k_values

if __name__ == '__main__':
    train = pd.read_csv('./data/train_V2.csv')
    # columns=['swimDistance']
    # max_corr_k_qcut(train,columns,10)
    