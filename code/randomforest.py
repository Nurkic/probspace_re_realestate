import optuna
import featuretools as ft
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sklearn.datasets
import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt

import re
import seaborn as sns

import seaborn as sns
from datetime import datetime, timezone, timedelta

from sklearn.decomposition import PCA

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

prep_train_label = im.Imputer(prep_train_label).num_imputer(
    ['TimeToNearestStation', 'Frontage', 'TotalFloorArea', 'BuildingYear', 'Breadth', 
    'CoverageRatio', 'FloorAreaRatio', 'BuildingAge', 'Area'])
prep_test_label = im.Imputer(prep_test_label).num_imputer(
    ['TimeToNearestStation', 'Frontage', 'TotalFloorArea', 'BuildingYear', 'Breadth', 
    'CoverageRatio', 'FloorAreaRatio', 'BuildingAge', 'Area'])

""" define data"""
train_X = prep_train_label.drop(["y", "id", "Prefecture", "Municipality"], axis=1)
train_y = np.log1p(prep_train_label["y"])
test_X = prep_test_label.drop(["id", "Prefecture", "Municipality"], axis=1)


""" target encoding"""
"""from feature_selection import FeatureSelector as FS, cross_validator
train_X_te, test_X_te = FS(train_X, train_y).target_encoder(test_X)"""

selected = {'Remarks', 'LandShape', 'BuildingAge', 'Use', 'TotalFloorArea', 'CoverageRatio', 'DistrictName', 
            'CityPlanning', 'Region', 'Direction', 'TimeToNearestStation', 'Purpose', 'Period', 'MunicipalityCode', 
            'NearestStation', 'Structure', 'Area', 'Renovation', 'BuildingYear', 'Frontage', 'Classification', 'Type', 
            'FloorAreaRatio', 'Breadth', 'era_name'}
#print(train_X.columns[np.isnan(train_X).any()])
#print(train_X.dtypes)
""" feature selection"""
#selected = FS(train_X, train_y).greedy_forward_selection()
#selected_te = FS(train_X_te, train_y).greedy_forward_selection()
#print("selected features:"+ str(selected))
#print("selected target encoding features:"+ str(selected_te))

def objective(trial):
    max_depth = trial.suggest_int('max_depth', 3, 32)
    max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 4, 64, 4))
    min_samples_split = trial.suggest_int("min_samples_split", 8, 16)
    
    FOLD_NUM = 4
    kf = KFold(n_splits=FOLD_NUM,
              #shuffle=True,
              random_state=42)
    scores = []
    feature_importance_df = pd.DataFrame()

    pred_cv = np.zeros(len(test.index))
    num_round = 10000

    


    for i, (tdx, vdx) in enumerate(kf.split(train_X[selected], train_y)):
        print(f'Fold : {i}')
        X_train, X_valid, y_train, y_valid = train_X[selected].iloc[tdx], train_X[selected].iloc[vdx], train_y.values[tdx], train_y.values[vdx]
        model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            max_leaf_nodes=max_leaf_nodes
            )
        model.fit(X_train.values, y_train)
        va_pred = model.predict(X_valid.values)
        va_pred[va_pred<0] = 0
        score_ = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(va_pred)))
        scores.append(score_)

    return np.mean(scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 結果の確認
print('Best trial:')
light_trial = study.best_trial

print('  Value: {}'.format(light_trial.value))

print('  Params: ')

