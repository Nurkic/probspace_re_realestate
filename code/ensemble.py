#import featuretools as ft
import lightgbm as lgb
#import optuna
import numpy as np
import sklearn.datasets
import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf
import xgboost as xgb
import re
import seaborn as sns
#from tensorflow import keras
#import keras.layers as L
import seaborn as sns
from datetime import datetime, timezone, timedelta
#from keras.models import Model
from sklearn.decomposition import PCA
#from keras import losses
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
import unicodedata

train_path = "../input/train_data.csv"
test_path = "../input/test_data.csv"

""" load raw data"""
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

""" Preprocessing"""
import preprocess as pr
import impute as im

import copy

df = train["y"]

predata = pd.concat([train.drop("y", axis=1), test], ignore_index=True)
predata_copy = copy.deepcopy(predata)
#predata_onehot = pr.Preprocessor(predata).all("onehot")
predata_label = pr.Preprocessor(predata_copy).all("label","nonpub")

prep_train_label = pd.concat([df, predata_label.iloc[:len(train), :]], axis=1)
prep_test_label = predata_label.iloc[len(train):, :]

""" define data"""
train_X = prep_train_label.drop(["y", "id", "Prefecture", "Municipality"], axis=1)
train_y = np.log1p(prep_train_label["y"])
test_X = prep_test_label.drop(["id", "Prefecture", "Municipality"], axis=1)


""" target encoding"""
"""from feature_selection import FeatureSelector as FS, cross_validator
train_X_te, test_X_te = FS(train_X, train_y).target_encoder(test_X)

print(test_X_te.head())"""



""" feature selection"""
#selected = FS(train_X, train_y).greedy_forward_selection()
"""selected_te = FS(train_X_te, train_y).greedy_forward_selection()
print("selected features:"+ str(selected))
print("selected target encoding features:"+ str(selected_te))"""

""" check cross validation score"""
"""cv1 = cross_validator(train_X, train_y)
cv2 = cross_validator(train_X[selected], train_y)
cv3 = cross_validator(train_X_te, train_y)
cv4 = cross_validator(train_X_te[selected_te], train_y)"""

"""print("base rmlse:"+ str(cv1))
print("feature_selected rmlse:"+ str(cv2))
print("target encoding rmlse:"+ str(cv3))
print("target encoding and feature selection rmlse:"+ str(cv4))"""

def rmsle(preds, data):
    y_true = data.get_label()
    y_pred = preds
    y_pred[y_pred<0] = 0
    y_true[y_true<0] = 0
    acc = np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))
    # name, result, is_higher_better
    return 'accuracy', acc, False

light_params = {'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        "seed":42,
        'learning_rate': 0.01,}
best_params =  {"learning_rate": 0.00596808175410754,
    "lambda_l1": 2.359343339881394e-08,
    "lambda_l2": 1.043244691067422e-06,
    "num_leaves": 221,
    "feature_fraction": 0.4272896136212019,
    "bagging_fraction": 0.4158670563729239,
    "bagging_freq": 1}
#best_params =  {}
light_params.update(best_params)

xgb_params = {'learning_rate': 0.1,
              'objective': 'reg:squarederror',
              'eval_metric': 'rmse',
              'seed': 42,
              'tree_method': 'hist'}
best_params = {"learning_rate": 0.0297902330982628,
    "lambda_l1": 0.0031603555384835144,
    "num_leaves": 219}
#best_params = {}
xgb_params.update(best_params)


FOLD_NUM = 11
kf = KFold(n_splits=FOLD_NUM,
           shuffle=True,
           random_state=42)
scores = []
feature_importance_df = pd.DataFrame()

pred_cv = np.zeros(len(test.index))
num_round = 10000

selected = {'Remarks', 'LandShape', 'BuildingAge', 'Use', 'TotalFloorArea', 'CoverageRatio',  'DistrictName', 
            'CityPlanning',  'Region', 'Direction', 'TimeToNearestStation', 'Period', 'MunicipalityCode', 
            'NearestStation', 'Structure', 'Area', 'BuildingYear', 'Frontage', 'Classification', 'Type', 
            'FloorAreaRatio', 'Breadth', 'Purpose','Renovation', 'era_name'}

object_cols = [
            'Type','Region','MunicipalityCode','DistrictName','NearestStation',
            'LandShape','Structure','Use','Classification','CityPlanning', 'Direction',
            'Remarks', 'Purpose','Renovation','era_name'
            ]

ymin = 0.1
ymax = 30000
n = FOLD_NUM
line = 7000


for i, (tdx, vdx) in enumerate(kf.split(train_X[selected], train_y)):
    print(f'Fold : {i}')
    ######LGB
    X_train, X_valid, y_train, y_valid = train_X[selected].iloc[tdx], train_X[selected].iloc[vdx], train_y.values[tdx], train_y.values[vdx]
    y_train2 = y_train.copy()
    y_train2[y_train2>=np.log1p(line)] = (y_train2[y_train2>=np.log1p(line)]-np.log1p(line))/n+np.log1p(line)
    # LGB
    X_train_lgb = copy.deepcopy(X_train)
    X_valid_lgb = copy.deepcopy(X_valid)
    for obj_col in object_cols:
        X_train_lgb[obj_col] = X_train_lgb[obj_col].astype("category")
        X_valid_lgb[obj_col] = X_valid_lgb[obj_col].astype("category")
    #lgb_train = lgb.Dataset(X_train, y_train)
    lgb_train = lgb.Dataset(X_train_lgb, np.clip(y_train2, ymin, np.log1p(ymax)))
    lgb_valid = lgb.Dataset(X_valid_lgb, y_valid)
    gbc = lgb.train(light_params, lgb_train, num_boost_round=num_round,
                  valid_names=["train", "valid"], valid_sets=[lgb_train, lgb_valid],
                  #feval=rmsle,
                  early_stopping_rounds=100, verbose_eval=500)
    if i ==0:
        importance_df = pd.DataFrame(gbc.feature_importance(), index=train_X[selected].columns, columns=['importance'])
    else:
        importance_df += pd.DataFrame(gbc.feature_importance(), index=train_X[selected].columns, columns=['importance'])
    gbc_va_pred = np.expm1(gbc.predict(X_valid_lgb, num_iteration=gbc.best_iteration))
    gbc_va_pred = np.clip(gbc_va_pred, 0, 100000)
    gbc_va_pred[gbc_va_pred<0] = 0

    # XGB
    #xgb_dataset = xgb.DMatrix(X_train, label=y_train)
    xgb_dataset = xgb.DMatrix(X_train, label=np.clip(y_train2, ymin, np.log1p(ymax)))
    xgb_test_dataset = xgb.DMatrix(X_valid, label=y_valid)
    xgbm = xgb.train(xgb_params, xgb_dataset, 10000, evals=[(xgb_dataset, 'train'),(xgb_test_dataset, 'eval')],
                      early_stopping_rounds=100, verbose_eval=500)
    xgbm_va_pred = np.expm1(xgbm.predict(xgb.DMatrix(X_valid)))
    xgbm_va_pred = np.clip(xgbm_va_pred, 0, 100000)
    xgbm_va_pred[xgbm_va_pred<0] = 0
    

    # ENS
    # lists for keep results
    lgb_xgb_rmsle = []
    lgb_xgb_alphas = []

    for alpha in np.linspace(0,1,101):
        y_pred = alpha*gbc_va_pred + (1 - alpha)*xgbm_va_pred
        #rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), y_pred))
        rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.where(np.expm1(y_valid)>5000, np.expm1(y_valid), y_pred)))
        lgb_xgb_rmsle.append(rmsle_score)
        lgb_xgb_alphas.append(alpha)
    
    lgb_xgb_rmsle = np.array(lgb_xgb_rmsle)
    lgb_xgb_alphas = np.array(lgb_xgb_alphas)

    lgb_xgb_best_alpha = lgb_xgb_alphas[np.argmin(lgb_xgb_rmsle)]

    print('best_rmsle=', lgb_xgb_rmsle.min())
    print('best_alpha=', lgb_xgb_best_alpha)
    plt.plot(lgb_xgb_alphas, lgb_xgb_rmsle)
    plt.title('f1_score for ensemble')
    plt.xlabel('alpha')
    plt.ylabel('f1_score')

    score_ = lgb_xgb_rmsle.min()
    scores.append(score_)

    #lgb_submission = np.expm1(gbc.predict((test_X[selected]), num_iteration=gbc.best_iteration))
    test_X_lgb = copy.deepcopy(test_X)
    for obj_col in object_cols:
        test_X_lgb[obj_col] = test_X_lgb[obj_col].astype("category")
    lgb_submission = np.clip(np.expm1(gbc.predict((test_X_lgb[selected]), num_iteration=gbc.best_iteration)), 0, 100000)
    lgb_submission[lgb_submission<0] = 0

    #xgbm_submission = np.expm1(xgbm.predict(xgb.DMatrix(test_X[selected])))
    xgbm_submission = np.clip(np.expm1(xgbm.predict(xgb.DMatrix(test_X[selected]))), 0, 100000)
    xgbm_submission[xgbm_submission<0] = 0

    submission = lgb_xgb_best_alpha*lgb_submission + (1 - lgb_xgb_best_alpha)*xgbm_submission

    pred_cv += submission/FOLD_NUM

print("##########")
print(np.mean(scores))

gbk_l = lgb.LGBMRegressor(n_estimators=200, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=1)
gbk_l.fit(train_X[train_y>np.log1p(5000)], np.log1p(train_y[train_y>np.log1p(5000)]))
y_pred_l = np.clip(np.expm1(gbk_l.predict(test_X)), 0, 100000)

""" export submit file"""
result = pd.DataFrame(test_X[selected].index, columns=["id"])
large_y_index = [1305, 1960, 8069]
pred_cv[large_y_index] = y_pred_l[large_y_index]
result["y"] = pred_cv
result.to_csv("../output/result_realestate_20200809_02.csv", index=False)