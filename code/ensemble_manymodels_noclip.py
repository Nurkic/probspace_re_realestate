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
test_X = prep_test_label.drop(["id", "Prefecture", "Municipality"], axis=1).reset_index(drop=True)

#print(train_X["Type"].value_counts())
""" divine data"""
train_X_tyuko = train_X[train_X["Type"]==1]
train_X_tatemono = train_X[train_X["Type"]==2]
train_X_toti = train_X[train_X["Type"]==3]
train_y_tyuko = train_y[train_X_tyuko.index]
train_y_tatemono = train_y[train_X_tatemono.index]
train_y_toti = train_y[train_X_toti.index]


""" train models"""

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


FOLD_NUM = 2
kf = KFold(n_splits=FOLD_NUM,
           shuffle=True,
           random_state=42)
scores = []
feature_importance_df = pd.DataFrame()

pred_cv = np.zeros(len(test.index))
pred_all = np.zeros(len(train.index))
num_round = 10000

selected = {'Remarks', 'LandShape', 'BuildingAge', 'Use', 'TotalFloorArea', 'CoverageRatio',  'DistrictName', 
            'CityPlanning',  'Region', 'Direction', 'TimeToNearestStation', 'Purpose', 'Period', 'MunicipalityCode', 
            'NearestStation', 'Structure', 'Area', 'Renovation', 'BuildingYear', 'Frontage', 'Classification', 'Type', 
            'FloorAreaRatio', 'Breadth', 'era_name'}

object_cols = [
            'Type','Region','MunicipalityCode', 'DistrictName','NearestStation',
            'LandShape','Structure','Use','Purpose','Classification','CityPlanning', 'Direction',
            'Renovation','Remarks','era_name'
            ]

train_X_lgb = copy.deepcopy(train_X)
for obj_col in object_cols:
    train_X_lgb[obj_col] = train_X_lgb[obj_col].astype("category")

ymin = 0.1
ymax = 30000
n = FOLD_NUM
line = 7000

""" all model"""
for i, (tdx, vdx) in enumerate(kf.split(train_X[selected], train_y)):
    print(f'Fold : {i}')
    ######LGB
    X_train, X_valid, y_train, y_valid = train_X[selected].iloc[tdx], train_X[selected].iloc[vdx], train_y.values[tdx], train_y.values[vdx]
    #y_train2 = y_train.copy()
    #y_train2[y_train2>=np.log1p(line)] = (y_train2[y_train2>=np.log1p(line)]-np.log1p(line))/n+np.log1p(line)
    # LGB
    #lgb_train = lgb.Dataset(X_train, y_train)
    X_train_lgb = copy.deepcopy(X_train)
    X_valid_lgb = copy.deepcopy(X_valid)
    for obj_col in object_cols:
        X_train_lgb[obj_col] = X_train_lgb[obj_col].astype("category")
        X_valid_lgb[obj_col] = X_valid_lgb[obj_col].astype("category")
    lgb_train = lgb.Dataset(X_train_lgb, y_train)
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
    #gbc_va_pred = np.clip(gbc_va_pred, 0, 100000)
    gbc_va_pred[gbc_va_pred<0] = 0

    # XGB
    #xgb_dataset = xgb.DMatrix(X_train, label=y_train)
    xgb_dataset = xgb.DMatrix(X_train, label=y_train)
    xgb_test_dataset = xgb.DMatrix(X_valid, label=y_valid)
    xgbm = xgb.train(xgb_params, xgb_dataset, 10000, evals=[(xgb_dataset, 'train'),(xgb_test_dataset, 'eval')],
                      early_stopping_rounds=100, verbose_eval=500)
    xgbm_va_pred = np.expm1(xgbm.predict(xgb.DMatrix(X_valid)))
    #xgbm_va_pred = np.clip(xgbm_va_pred, 0, 100000)
    xgbm_va_pred[xgbm_va_pred<0] = 0
    

    # ENS
    # lists for keep results
    lgb_xgb_rmsle = []
    lgb_xgb_alphas = []

    for alpha in np.linspace(0,1,101):
        y_pred = alpha*gbc_va_pred + (1 - alpha)*xgbm_va_pred
        #rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), y_pred))
        rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), y_pred))
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
    lgb_submission = np.expm1(gbc.predict(test_X_lgb[selected], num_iteration=gbc.best_iteration))
    lgb_submission[lgb_submission<0] = 0

    #xgbm_submission = np.expm1(xgbm.predict(xgb.DMatrix(test_X[selected])))
    xgbm_submission = np.expm1(xgbm.predict(xgb.DMatrix(test_X[selected])))
    xgbm_submission[xgbm_submission<0] = 0

    submission = lgb_xgb_best_alpha*lgb_submission + (1 - lgb_xgb_best_alpha)*xgbm_submission

    pred_cv += submission/FOLD_NUM

    lgb_pred = np.expm1(gbc.predict(train_X_lgb[selected], num_iteration=gbc.best_iteration))
    xgbm_pred = np.expm1(xgbm.predict(xgb.DMatrix(train_X[selected])))

    submission = lgb_xgb_best_alpha*lgb_pred + (1 - lgb_xgb_best_alpha)*xgbm_pred

    pred_all += submission/FOLD_NUM

print("##########")
print(np.mean(scores))

scores = []
feature_importance_df = pd.DataFrame()

pred_cv_tyuko = np.zeros(len(test.index))
pred_tyuko = np.zeros(len(train.index))

""" tyuko model"""
FOLD_NUM = 2
for i, (tdx, vdx) in enumerate(kf.split(train_X_tyuko[selected], train_y_tyuko)):
    print(f'Fold : {i}')
    ######LGB
    X_train, X_valid, y_train, y_valid = train_X_tyuko[selected].iloc[tdx], train_X_tyuko[selected].iloc[vdx], train_y_tyuko.values[tdx], train_y_tyuko.values[vdx]
    #y_train2 = y_train.copy()
    #y_train2[y_train2>=np.log1p(line)] = (y_train2[y_train2>=np.log1p(line)]-np.log1p(line))/n+np.log1p(line)
    # LGB
    #lgb_train = lgb.Dataset(X_train, y_train)
    X_train_lgb = copy.deepcopy(X_train)
    X_valid_lgb = copy.deepcopy(X_valid)
    for obj_col in object_cols:
        X_train_lgb[obj_col] = X_train_lgb[obj_col].astype("category")
        X_valid_lgb[obj_col] = X_valid_lgb[obj_col].astype("category")
    lgb_train = lgb.Dataset(X_train_lgb, y_train)
    lgb_valid = lgb.Dataset(X_valid_lgb, y_valid)
    gbc = lgb.train(light_params, lgb_train, num_boost_round=num_round,
                  valid_names=["train", "valid"], valid_sets=[lgb_train, lgb_valid],
                  #feval=rmsle,
                  early_stopping_rounds=100, verbose_eval=500)
    
    gbc_va_pred_tyuko = np.expm1(gbc.predict(X_valid_lgb, num_iteration=gbc.best_iteration))
    #gbc_va_pred_tyuko = np.clip(gbc_va_pred_tyuko, 0, 100000)
    gbc_va_pred_tyuko[gbc_va_pred_tyuko<0] = 0

    # XGB
    #xgb_dataset = xgb.DMatrix(X_train, label=y_train)
    xgb_dataset = xgb.DMatrix(X_train, label=y_train)
    xgb_test_dataset = xgb.DMatrix(X_valid, label=y_valid)
    xgbm = xgb.train(xgb_params, xgb_dataset, 10000, evals=[(xgb_dataset, 'train'),(xgb_test_dataset, 'eval')],
                      early_stopping_rounds=100, verbose_eval=500)
    xgbm_va_pred_tyuko = np.expm1(xgbm.predict(xgb.DMatrix(X_valid)))
    #xgbm_va_pred_tyuko = np.clip(xgbm_va_pred_tyuko, 0, 100000)
    xgbm_va_pred_tyuko[xgbm_va_pred_tyuko<0] = 0
    

    # ENS
    # lists for keep results
    lgb_xgb_rmsle = []
    lgb_xgb_alphas = []

    for alpha in np.linspace(0,1,101):
        y_pred = alpha*gbc_va_pred_tyuko + (1 - alpha)*xgbm_va_pred_tyuko
        #rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), y_pred))
        rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), y_pred))
        lgb_xgb_rmsle.append(rmsle_score)
        lgb_xgb_alphas.append(alpha)
    
    lgb_xgb_rmsle = np.array(lgb_xgb_rmsle)
    lgb_xgb_alphas = np.array(lgb_xgb_alphas)

    lgb_xgb_best_alpha_tyuko = lgb_xgb_alphas[np.argmin(lgb_xgb_rmsle)]

    print('best_rmsle=', lgb_xgb_rmsle.min())
    print('best_alpha=', lgb_xgb_best_alpha_tyuko)
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
    lgb_submission = np.expm1(gbc.predict(test_X_lgb[selected], num_iteration=gbc.best_iteration))
    lgb_submission[lgb_submission<0] = 0

    #xgbm_submission = np.expm1(xgbm.predict(xgb.DMatrix(test_X[selected])))
    xgbm_submission = np.expm1(xgbm.predict(xgb.DMatrix(test_X[selected])))
    xgbm_submission[xgbm_submission<0] = 0

    submission = lgb_xgb_best_alpha_tyuko*lgb_submission + (1 - lgb_xgb_best_alpha_tyuko)*xgbm_submission

    pred_cv_tyuko += submission/FOLD_NUM

    lgb_pred = np.expm1(gbc.predict(train_X_lgb[selected], num_iteration=gbc.best_iteration))
    xgbm_pred = np.expm1(xgbm.predict(xgb.DMatrix(train_X[selected])))

    submission = lgb_xgb_best_alpha_tyuko*lgb_pred + (1 - lgb_xgb_best_alpha_tyuko)*xgbm_pred

    pred_tyuko += submission/FOLD_NUM

print("##########")
print(np.mean(scores))

scores = []
feature_importance_df = pd.DataFrame()

pred_cv_tatemono = np.zeros(len(test.index))
pred_tatemono = np.zeros(len(train.index))
""" tatemono model"""
for i, (tdx, vdx) in enumerate(kf.split(train_X_tatemono[selected], train_y_tatemono)):
    print(f'Fold : {i}')
    ######LGB
    X_train, X_valid, y_train, y_valid = train_X_tatemono[selected].iloc[tdx], train_X_tatemono[selected].iloc[vdx], train_y_tatemono.values[tdx], train_y_tatemono.values[vdx]
    #y_train2 = y_train.copy()
    #y_train2[y_train2>=np.log1p(line)] = (y_train2[y_train2>=np.log1p(line)]-np.log1p(line))/n+np.log1p(line)
    # LGB
    #lgb_train = lgb.Dataset(X_train, y_train)
    X_train_lgb = copy.deepcopy(X_train)
    X_valid_lgb = copy.deepcopy(X_valid)
    for obj_col in object_cols:
        X_train_lgb[obj_col] = X_train_lgb[obj_col].astype("category")
        X_valid_lgb[obj_col] = X_valid_lgb[obj_col].astype("category")
    lgb_train = lgb.Dataset(X_train_lgb, y_train)
    lgb_valid = lgb.Dataset(X_valid_lgb, y_valid)
    gbc = lgb.train(light_params, lgb_train, num_boost_round=num_round,
                  valid_names=["train", "valid"], valid_sets=[lgb_train, lgb_valid],
                  #feval=rmsle,
                  early_stopping_rounds=100, verbose_eval=500)
    
    gbc_va_pred_tatemono = np.expm1(gbc.predict(X_valid_lgb, num_iteration=gbc.best_iteration))
    #gbc_va_pred_tatemono = np.clip(gbc_va_pred_tatemono, 0, 100000)
    gbc_va_pred_tatemono[gbc_va_pred_tatemono<0] = 0

    # XGB
    #xgb_dataset = xgb.DMatrix(X_train, label=y_train)
    xgb_dataset = xgb.DMatrix(X_train, label=y_train)
    xgb_test_dataset = xgb.DMatrix(X_valid, label=y_valid)
    xgbm = xgb.train(xgb_params, xgb_dataset, 10000, evals=[(xgb_dataset, 'train'),(xgb_test_dataset, 'eval')],
                      early_stopping_rounds=100, verbose_eval=500)
    xgbm_va_pred_tatemono = np.expm1(xgbm.predict(xgb.DMatrix(X_valid)))
    #xgbm_va_pred_tatemono = np.clip(xgbm_va_pred_tatemono, 0, 100000)
    xgbm_va_pred_tatemono[xgbm_va_pred_tatemono<0] = 0
    

    # ENS
    # lists for keep results
    lgb_xgb_rmsle = []
    lgb_xgb_alphas = []

    for alpha in np.linspace(0,1,101):
        y_pred = alpha*gbc_va_pred_tatemono + (1 - alpha)*xgbm_va_pred_tatemono
        #rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), y_pred))
        rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), y_pred))
        lgb_xgb_rmsle.append(rmsle_score)
        lgb_xgb_alphas.append(alpha)
    
    lgb_xgb_rmsle = np.array(lgb_xgb_rmsle)
    lgb_xgb_alphas = np.array(lgb_xgb_alphas)

    lgb_xgb_best_alpha_tatemono = lgb_xgb_alphas[np.argmin(lgb_xgb_rmsle)]

    print('best_rmsle=', lgb_xgb_rmsle.min())
    print('best_alpha=', lgb_xgb_best_alpha_tatemono)
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
    lgb_submission = np.expm1(gbc.predict(test_X_lgb[selected], num_iteration=gbc.best_iteration))
    lgb_submission[lgb_submission<0] = 0

    #xgbm_submission = np.expm1(xgbm.predict(xgb.DMatrix(test_X[selected])))
    xgbm_submission = np.expm1(xgbm.predict(xgb.DMatrix(test_X[selected])))
    xgbm_submission[xgbm_submission<0] = 0

    submission = lgb_xgb_best_alpha_tatemono*lgb_submission + (1 - lgb_xgb_best_alpha_tatemono)*xgbm_submission

    pred_cv_tatemono += submission/FOLD_NUM

    lgb_pred = np.expm1(gbc.predict(train_X_lgb[selected], num_iteration=gbc.best_iteration))
    xgbm_pred = np.expm1(xgbm.predict(xgb.DMatrix(train_X[selected])))

    submission = lgb_xgb_best_alpha_tatemono*lgb_pred + (1 - lgb_xgb_best_alpha_tatemono)*xgbm_pred

    pred_tatemono += submission/FOLD_NUM

print("##########")
print(np.mean(scores))

scores = []
feature_importance_df = pd.DataFrame()

pred_cv_toti = np.zeros(len(test.index))
pred_toti = np.zeros(len(train.index))
""" toti model"""
for i, (tdx, vdx) in enumerate(kf.split(train_X_toti[selected], train_y_toti)):
    print(f'Fold : {i}')
    ######LGB
    X_train, X_valid, y_train, y_valid = train_X_toti[selected].iloc[tdx], train_X_toti[selected].iloc[vdx], train_y_toti.values[tdx], train_y_toti.values[vdx]
    #y_train2 = y_train.copy()
    #y_train2[y_train2>=np.log1p(line)] = (y_train2[y_train2>=np.log1p(line)]-np.log1p(line))/n+np.log1p(line)
    # LGB
    #lgb_train = lgb.Dataset(X_train, y_train)
    X_train_lgb = copy.deepcopy(X_train)
    X_valid_lgb = copy.deepcopy(X_valid)
    for obj_col in object_cols:
        X_train_lgb[obj_col] = X_train_lgb[obj_col].astype("category")
        X_valid_lgb[obj_col] = X_valid_lgb[obj_col].astype("category")
    lgb_train = lgb.Dataset(X_train_lgb, y_train)
    lgb_valid = lgb.Dataset(X_valid_lgb, y_valid)
    gbc = lgb.train(light_params, lgb_train, num_boost_round=num_round,
                  valid_names=["train", "valid"], valid_sets=[lgb_train, lgb_valid],
                  #feval=rmsle,
                  early_stopping_rounds=100, verbose_eval=500)
    
    gbc_va_pred_toti = np.expm1(gbc.predict(X_valid_lgb, num_iteration=gbc.best_iteration))
    #gbc_va_pred_toti = np.clip(gbc_va_pred_toti, 0, 100000)
    gbc_va_pred_toti[gbc_va_pred_toti<0] = 0

    # XGB
    #xgb_dataset = xgb.DMatrix(X_train, label=y_train)
    xgb_dataset = xgb.DMatrix(X_train, label=y_train)
    xgb_test_dataset = xgb.DMatrix(X_valid, label=y_valid)
    xgbm = xgb.train(xgb_params, xgb_dataset, 10000, evals=[(xgb_dataset, 'train'),(xgb_test_dataset, 'eval')],
                      early_stopping_rounds=100, verbose_eval=500)
    xgbm_va_pred_toti = np.expm1(xgbm.predict(xgb.DMatrix(X_valid)))
    #xgbm_va_pred_toti = np.clip(xgbm_va_pred_toti, 0, 100000)
    xgbm_va_pred_toti[xgbm_va_pred_toti<0] = 0
    

    # ENS
    # lists for keep results
    lgb_xgb_rmsle = []
    lgb_xgb_alphas = []

    for alpha in np.linspace(0,1,101):
        y_pred = alpha*gbc_va_pred_toti + (1 - alpha)*xgbm_va_pred_toti
        #rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), y_pred))
        rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), y_pred))
        lgb_xgb_rmsle.append(rmsle_score)
        lgb_xgb_alphas.append(alpha)
    
    lgb_xgb_rmsle = np.array(lgb_xgb_rmsle)
    lgb_xgb_alphas = np.array(lgb_xgb_alphas)

    lgb_xgb_best_alpha_toti = lgb_xgb_alphas[np.argmin(lgb_xgb_rmsle)]

    print('best_rmsle=', lgb_xgb_rmsle.min())
    print('best_alpha=', lgb_xgb_best_alpha_toti)
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
    lgb_submission = np.expm1(gbc.predict(test_X_lgb[selected], num_iteration=gbc.best_iteration))
    lgb_submission[lgb_submission<0] = 0

    #xgbm_submission = np.expm1(xgbm.predict(xgb.DMatrix(test_X[selected])))
    xgbm_submission = np.expm1(xgbm.predict(xgb.DMatrix(test_X[selected])))
    xgbm_submission[xgbm_submission<0] = 0

    submission = lgb_xgb_best_alpha_toti*lgb_submission + (1 - lgb_xgb_best_alpha_toti)*xgbm_submission

    pred_cv_toti += submission/FOLD_NUM

    lgb_pred = np.expm1(gbc.predict(train_X_lgb[selected], num_iteration=gbc.best_iteration))
    xgbm_pred = np.expm1(xgbm.predict(xgb.DMatrix(train_X[selected])))

    submission = lgb_xgb_best_alpha_toti*lgb_pred + (1 - lgb_xgb_best_alpha_toti)*xgbm_pred

    pred_toti += submission/FOLD_NUM

print("##########")
print(np.mean(scores))

"""pred_cv_models = copy.deepcopy(pred_cv)
pred_cv_models[train_X_tyuko.index] = pred_cv_tyuko[train_X_tyuko.index]
pred_cv_models[train_X_tatemono.index] = pred_cv_tatemono[train_X_tatemono.index]
pred_cv_models[train_X_toti.index] = pred_cv_toti[train_X_toti.index]"""

""" bagging"""
y_pred_all = pred_all
y_pred_tyuko = pred_tyuko
y_pred_tatemono = pred_tatemono
y_pred_toti = pred_toti
lgb_xgb_rmsle = []
lgb_xgb_alphas = []
for alpha in np.linspace(0,1,101):
    y_pred = alpha*y_pred_all[train_X_tyuko.index] + (1 - alpha)*y_pred_tyuko[train_X_tyuko.index]
    #rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), y_pred))
    rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(train_y[train_X_tyuko.index]), y_pred))
    lgb_xgb_rmsle.append(rmsle_score)
    lgb_xgb_alphas.append(alpha)

lgb_xgb_rmsle = np.array(lgb_xgb_rmsle)
lgb_xgb_alphas = np.array(lgb_xgb_alphas)

lgb_xgb_best_alpha_tyuko = lgb_xgb_alphas[np.argmin(lgb_xgb_rmsle)]
y_pred_all[train_X_tyuko.index] = lgb_xgb_best_alpha_tyuko*y_pred_all[train_X_tyuko.index] + (1 - lgb_xgb_best_alpha_tyuko)*y_pred_tyuko[train_X_tyuko.index]
pred_cv[test_X[test_X["Type"]==1].index] = lgb_xgb_best_alpha_tyuko*pred_cv[test_X[test_X["Type"]==1].index] + (1 - lgb_xgb_best_alpha_tyuko)*pred_cv_tyuko[test_X[test_X["Type"]==1].index]

lgb_xgb_rmsle = []
lgb_xgb_alphas = []
for alpha in np.linspace(0,1,101):
    y_pred = alpha*y_pred_all[train_X_tatemono.index] + (1 - alpha)*y_pred_tyuko[train_X_tatemono.index]
    #rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), y_pred))
    rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(train_y[train_X_tatemono.index]), y_pred))
    lgb_xgb_rmsle.append(rmsle_score)
    lgb_xgb_alphas.append(alpha)

lgb_xgb_rmsle = np.array(lgb_xgb_rmsle)
lgb_xgb_alphas = np.array(lgb_xgb_alphas)

lgb_xgb_best_alpha_tatemono = lgb_xgb_alphas[np.argmin(lgb_xgb_rmsle)]
y_pred_all[train_X_tatemono.index] = lgb_xgb_best_alpha_tatemono*y_pred_all[train_X_tatemono.index] + (1 - lgb_xgb_best_alpha_tatemono)*y_pred_tatemono[train_X_tatemono.index]
pred_cv[test_X[test_X["Type"]==2].index] = lgb_xgb_best_alpha_tatemono*pred_cv[test_X[test_X["Type"]==2].index] + (1 - lgb_xgb_best_alpha_tatemono)*pred_cv_tatemono[test_X[test_X["Type"]==2].index]

lgb_xgb_rmsle = []
lgb_xgb_alphas = []
for alpha in np.linspace(0,1,101):
    y_pred = alpha*y_pred_all[train_X_toti.index] + (1 - alpha)*y_pred_toti[train_X_toti.index]
    #rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), y_pred))
    rmsle_score = np.sqrt(mean_squared_log_error(np.expm1(train_y[train_X_toti.index]), y_pred))
    lgb_xgb_rmsle.append(rmsle_score)
    lgb_xgb_alphas.append(alpha)

lgb_xgb_rmsle = np.array(lgb_xgb_rmsle)
lgb_xgb_alphas = np.array(lgb_xgb_alphas)

lgb_xgb_best_alpha_toti = lgb_xgb_alphas[np.argmin(lgb_xgb_rmsle)]
y_pred_all[train_X_toti.index] = lgb_xgb_best_alpha_toti*y_pred_all[train_X_toti.index] + (1 - lgb_xgb_best_alpha_toti)*y_pred_toti[train_X_toti.index]
pred_cv[test_X[test_X["Type"]==3].index] = lgb_xgb_best_alpha_toti*pred_cv[test_X[test_X["Type"]==3].index] + (1 - lgb_xgb_best_alpha_toti)*pred_cv_toti[test_X[test_X["Type"]==3].index]

y_pred_all[y_pred_all<0] = 0

print(np.sqrt(mean_squared_log_error(np.expm1(train_y), y_pred_all)))

"""gbk_l = lgb.LGBMRegressor(n_estimators=200, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=1)
gbk_l.fit(train_X[train_y>np.log1p(5000)], np.log1p(train_y[train_y>np.log1p(5000)]))
y_pred_l = np.clip(np.expm1(gbk_l.predict(test_X)), 0, 100000)"""

""" export submit file"""
result = pd.DataFrame(test_X[selected].index, columns=["id"])
"""large_y_index = [1305, 1960, 8069]
pred_cv[large_y_index] = y_pred_l[large_y_index]"""
result["y"] = pred_cv
result.to_csv("../output/result_realestate_20200809_04.csv", index=False)