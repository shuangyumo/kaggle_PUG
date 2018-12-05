import pandas as pd
from continues_to_discrete import *
def totalDistance(df):
    '''

    :param df:
    :param columns: list,
    :return: 新的df,增加了一列总距离。之前的三个距离删掉
    '''
    data=df.copy()
    corr=df[['walkDistance','winPlacePerc']].corr().iloc[0,1]
    walk=1
    ride=0
    swim=0
    for i in range(10):
        for j in range(10):
            for k in range(10):
                data['totalDistance']=data['walkDistance']*i/10+data['rideDistance']*j/10+\
                                      data['swimDistance']*k/10
                temp_corr=data[['totalDistance','winPlacePerc']].corr().iloc[0,1]
                #### 如果当前相关性大于最大相关性，更新相应的系数
                if temp_corr>corr:
                    corr=temp_corr
                    walk=i
                    ride=j
                    swim=k
                    print('corr: {} walk: {} ride: {}+ swim: {}'.format(corr,walk,ride,swim))
    data['totalDistance']=data['walkDistance']*walk/10+data['rideDistance']*ride/10+\
                                      data['swimDistance']*swim/10
    data.drop(columns=['rideDistance','walkDistance','swimDistance'],inplace=True,axis=1)
    return data
if __name__ == '__main__':
    train=pd.read_csv('../data/train_V2.csv')
    # train=totalDistance(train)
    # print(train['totalDistance'])
    train['totalDistance']=0.9*train['walkDistance']+0.7*train['swimDistance']+train['rideDistance']*0.1
    #####最终的结果 corr: 0.8165787202176797 walk: 9 ride: 1  swim: 7
    train=max_corr_qcut(train,['totalDistance','walkDistance'],40,30)
    print(train[['winPlacePerc','totalDistanceQCategory','walkDistanceQCategory']].corr())