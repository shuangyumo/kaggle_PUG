from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
import numpy as np
import pandas as pd
import os
import gc
# 别人的自定义损失函数,在parameter里面：object里面赋值
def custom_loss(y_true,y_pred):
    penalty=2.0
    grad=-y_true/y_pred+penalty*(1-y_true)/(1-y_pred) #梯度
    hess=y_true/(y_pred**2)+penalty*(1-y_true)/(1-y_pred)**2 #2阶导
    return grad,hess
# 自定义评价函数
def mse(y_pred,dtrain): #preds是结果（概率值），dtrain是个带label的DMatrix
    labels=dtrain.get_label() #提取label
    t=mean_absolute_error(labels, y_pred)
    print(t)
    return 'mse',t



def get_XgbRegressor(train_data,train_target,test_data,feature_names,parameters,early_stopping_rounds,num_folds,eval_metric,model_name='model',stratified=False):
    '''
    :param train_data: 一定是numpy
    :param train_target:
    :param parameters:
    :param round:
    :param k:
    :param eval_metrics:自定义 or 内置字符串
    :return:
    '''
    reg=XGBRegressor()
    reg.set_params(**parameters)

    # 定义一些变量
    oof_preds = np.zeros((train_data.shape[0],))
    sub_preds = np.zeros((test_data.shape[0],))
    feature_importance_df = pd.DataFrame()
    cv_result = []

    # K-flod
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1234)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1234)

    for n_flod, (train_index, val_index) in enumerate(folds.split(train_data, train_target)):
        train_X=train_data[train_index]
        val_X=train_data[val_index]
        train_Y=train_target[train_index]
        val_Y=train_target[val_index]
        # 参数初步定之后划分20%为验证集，准备一个watchlist 给train和validation set ,设置num_round 足够大（比如100000），以至于你能发现每一个round 的验证集预测结果，
        # 如果在某一个round后 validation set 的预测误差上升了，你就可以停止掉正在运行的程序了。
        watchlist= [(train_X, train_Y), (val_X, val_Y)]

        # early_stop 看validate的eval是否下降，这时候必须传eval_set,并取eval_set的最后一个作为validate
        reg.fit(train_X,train_Y,early_stopping_rounds=early_stopping_rounds, eval_set=watchlist,eval_metric=eval_metric)

       # 获得每次的预测值补充
        oof_preds[val_index]=reg.predict(val_X)
        # 获得预测的平均值，这里直接加完再除m
        sub_preds+= reg.predict(test_data)
        result = mean_absolute_error(val_Y, reg.predict(val_X))
        print('Fold %2d macro-f1 : %.6f' % (n_flod + 1, result))
        cv_result.append(round(result,5))
        gc.collect()
        # 默认就是gain 如果要修改要再参数定义中修改importance_type
        # 保存特征重要度
        gain = reg.feature_importances_
        fold_importance_df = pd.DataFrame({'feature': feature_names,
                                           'gain': 100 * gain / gain.sum(),
                                           'fold': n_flod,
                                           }).sort_values('gain', ascending=False)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    if not os.path.isdir('./sub'):
        os.makedirs('./sub')
    pd.DataFrame(oof_preds,columns=['class']).to_csv('./sub/val_{}.csv'.format(model_name), index=False)
    pd.DataFrame(sub_preds, columns=['class']).to_csv('./sub/test_{}.csv'.format(model_name), index=False)
    print('cv_result', cv_result)

    save_importances(feature_importance_df, model_name)
    return reg

def save_importances(feature_importance_df,model_name):
    if not os.path.isdir('./feature_importance'):
        os.makedirs('./feature_importance')
    ft = feature_importance_df[["feature", "gain"]].groupby("feature").mean().sort_values(by="gain",ascending=False)
    ft.to_csv('./feature_importance/importance_lightgbm_{}.csv'.format(model_name), index=True)
