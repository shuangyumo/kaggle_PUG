import pandas as pd
import numpy as np
import decimal

####等频切分成K份
def continues_qcut(df, columns, k_values):
    '''

    :param df: dataframe
    :param columns: list
    :param k_values: {} ，与columns 相对应的切分k值，column 为key,值为切分的k值
    :return: df,没有删除之前的列
    '''
    data = df.copy()
    for column in columns:
        k = k_values[column]
        num = k_values[column]
        for i in range(k):
            ####因为切分的时候要去重，这里是计算去重后的标签值
            if data[column].quantile(i / k) == data[column].quantile((i + 1) / k):
                num = num - 1
        columnName = column + 'QCategory'
        data[columnName] = pd.qcut(data[column], k, labels=[i for i in range(num)], duplicates='drop')
        data[columnName] = data[columnName].to_frame()
        # corr = data[['winPlacePerc', columnName]].corr()
        # print(corr)
    return data

def continues_cut(df,columns,k_values):
    '''

    :param df: dataframe
    :param columns:  list
    :param k_values: {} ，与columns 相对应的切分k值，column 为key,值为切分的k值
    :return: df,没有删除之前的列
    '''
    data = df.copy()
    for column in columns:
        k = k_values[column]
        columnName = column + 'Category'
        data[columnName] = pd.cut(data[column], k, labels=[i for i in range(k)], duplicates='drop')
        data[columnName] = data[columnName].to_frame()
        # print(data[columnName])
        # corr = data[['winPlacePerc', columnName]].corr()
        # print(corr)
    return data####kmax 是K值的上界,columns是传入的list列名
def max_corr_qcut(df, columns, kmax,kmin=1):
    '''

    :param df: dataframe
    :param columns: list
    :param kmax: 切分K的最大值
    :param kmin: k的最小值
    :return: 新的df,删除了之前的列
    '''
    k_values = {}
    k_cut_values = {}
    corr_values = {}
    for column in columns:
        k_values[column] = 0
        k_cut_values[column] = 0
        corr_values[column] = 0
    for k in range(kmin, kmax + 1):
        for column in columns:
            k_cut_values[column] = k
        data = continues_qcut(df, columns, k_cut_values)
        for column in columns:
            corr = data[[column + 'QCategory', 'winPlacePerc']].corr().iloc[0, 1]
            ####如果corr绝对值变大，更新corr_value 和 k 值
            if corr_values[column] ** 2 < corr ** 2:
                corr_values[column] = corr
                k_values[column] = k
    ####用最优的K值切分，
    print(k_values)

    all = continues_qcut(df, columns, k_values)
    all.drop(columns, axis=1, inplace=True)
    return all
def max_corr_cut(df, columns, kmax,kmin=1):
    '''

    :param df: dataframe
    :param columns: list
    :param kmax: 切分K的最大值
    :param kmin: k的最小值
    :return: 新的df,删除了之前的列
    '''
    k_values = {}
    k_cut_values = {}
    corr_values = {}
    for column in columns:
        k_values[column] = 0
        k_cut_values[column] = 0
        corr_values[column] = 0
    for k in range(kmin, kmax + 1):
        for column in columns:
            k_cut_values[column] = k
        data = continues_cut(df, columns, k_cut_values)
        for column in columns:
            corr = data[[column + 'QCategory', 'winPlacePerc']].corr().iloc[0, 1]
            ####如果corr绝对值变大，更新corr_value 和 k 值
            if corr_values[column] ** 2 < corr ** 2:
                corr_values[column] = corr
                k_values[column] = k
    ####用最优的K值切分，
    print(k_values)
    all = continues_cut(df, columns, k_values)
    all.drop(columns, axis=1, inplace=True)
    return all
if __name__ == '__main__':
    train = pd.read_csv('./data/train_V2.csv')
    # columns=['swimDistance']
    # max_corr_k_qcut(train,columns,10)
    