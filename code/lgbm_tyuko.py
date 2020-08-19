import optuna
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
predata_label = pr.Preprocessor(predata_copy).all("label", "nonpub")

prep_train_label = pd.concat([df, predata_label.iloc[:len(train), :]], axis=1)
prep_test_label = predata_label.iloc[len(train):, :]

""" define data"""
train_X = prep_train_label.drop(["y", "id", "Prefecture", "Municipality"], axis=1)
train_y = np.log1p(prep_train_label["y"])
test_X = prep_test_label.drop(["id", "Prefecture", "Municipality"], axis=1)

""" divine data"""
train_X_tyuko = train_X[train_X["Type"]==1]
train_X_tatemono = train_X[train_X["Type"]==2]
train_X_toti = train_X[train_X["Type"]==3]
train_y_tyuko = train_y[train_X_tyuko.index]
train_y_tatemono = train_y[train_X_tatemono.index]
train_y_toti = train_y[train_X_toti.index]

""" target encoding"""
"""from feature_selection import FeatureSelector as FS, cross_validator
train_X_te, test_X_te = FS(train_X, train_y).target_encoder(test_X)"""

selected = {'Remarks', 'LandShape', 'BuildingAge', 'Use', 'TotalFloorArea', 'CoverageRatio', 'DistrictName', 
            'CityPlanning', 'Region', 'Direction', 'TimeToNearestStation', 'Purpose', 'Period', 'MunicipalityCode', 
            'NearestStation', 'Structure', 'Area', 'Renovation', 'BuildingYear', 'Frontage', 'Classification', 'Type', 
            'FloorAreaRatio', 'Breadth', 'era_name'}

object_cols = [
            'Type','Region','MunicipalityCode', 'DistrictName','NearestStation',
            'LandShape','Structure','Use','Purpose','Classification','CityPlanning', 'Direction',
            'Renovation','Remarks','era_name'
            ]

""" feature selection"""
#selected = FS(train_X, train_y).greedy_forward_selection()
#selected_te = FS(train_X_te, train_y).greedy_forward_selection()
#print("selected features:"+ str(selected))
#print("selected target encoding features:"+ str(selected_te))


def objective(trial):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        "seed":42,
        "learning_rate":trial.suggest_loguniform('learning_rate', 0.005, 0.03),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    }
    FOLD_NUM = 4
    kf = KFold(n_splits=FOLD_NUM,
              #shuffle=True,
              random_state=42)
    scores = []
    feature_importance_df = pd.DataFrame()

    pred_cv = np.zeros(len(test.index))
    num_round = 10000
    
    ymin = 0.1
    ymax = 30000
    n = FOLD_NUM
    line = 7000
    


    for i, (tdx, vdx) in enumerate(kf.split(train_X_tyuko[selected], train_y_tyuko)):
        print(f'Fold : {i}')
        X_train, X_valid, y_train, y_valid = train_X_tyuko[selected].iloc[tdx], train_X_tyuko[selected].iloc[vdx], train_y_tyuko.values[tdx], train_y_tyuko.values[vdx]
        y_train2 = y_train.copy()
        y_train2[y_train2>=np.log1p(line)] = (y_train2[y_train2>=np.log1p(line)]-np.log1p(line))/n+np.log1p(line)
        #lgb_train = lgb.Dataset(X_train, y_train)
        X_train_lgb = copy.deepcopy(X_train)
        X_valid_lgb = copy.deepcopy(X_valid)
        for obj_col in object_cols:
            X_train_lgb[obj_col] = X_train_lgb[obj_col].astype("category")
            X_valid_lgb[obj_col] = X_valid_lgb[obj_col].astype("category")
        lgb_train = lgb.Dataset(X_train_lgb, np.clip(y_train2, ymin, np.log1p(ymax)))
        lgb_valid = lgb.Dataset(X_valid_lgb, y_valid)
        model = lgb.train(params, lgb_train, num_boost_round=num_round,
                      valid_names=["train", "valid"], valid_sets=[lgb_train, lgb_valid],
                      early_stopping_rounds=50, verbose_eval=10000)

        va_pred = np.expm1(model.predict(X_valid_lgb, num_iteration=model.best_iteration))
        va_pred = np.clip(va_pred, 0, 100000)
        va_pred[va_pred<0] = 0
        score_ = np.sqrt(mean_squared_log_error(np.expm1(y_valid), va_pred))
        scores.append(score_)

    return np.mean(scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 結果の確認
print('Best trial:')
light_trial = study.best_trial

print('  Value: {}'.format(light_trial.value))

print('  Params: ')

with open("lightgbmparams.txt", "w") as file:
    for key, value in light_trial.params.items():
       print('    "{}": {},'.format(key, value))
       file.write('"{}": {},'.format(key, value))
