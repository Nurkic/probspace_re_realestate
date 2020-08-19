import optuna
import featuretools as ft
from sklearn import linear_model
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
predata_onehot = pr.Preprocessor(predata_copy).all("onehot", "nonpub")
#predata_label = pr.Preprocessor(predata_copy).all("label", "nonpub")

#prep_train_label = pd.concat([df, predata_label.iloc[:len(train), :]], axis=1)
#prep_test_label = predata_label.iloc[len(train):, :]

num_list = [
    "TimeToNearestStation", "TotalFloorArea", "Area", "Frontage", "BuildingYear", "BuildingAge", 
    "Breadth", "CoverageRatio", "FloorAreaRatio", "Period"
    ]
predata_onehot = im.Imputer(predata_onehot).num_imputer(num_list)
print(predata_onehot[num_list].isnull().sum( ))

prep_train_onehot = pd.concat([df, predata_onehot.iloc[:len(train), :]], axis=1)
prep_test_onehot = predata_onehot.iloc[len(train):, :]

""" define data"""
train_X = prep_train_onehot.drop(["y", "id", "Prefecture", "Municipality"], axis=1)
train_y = np.log1p(df)
test_X = prep_test_onehot.drop(["id", "Prefecture", "Municipality"], axis=1)

print(train_X.dtypes)
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
    alpha = trial.suggest_loguniform('alpha', 0.01, 0.99)
    l1_ratio = int(trial.suggest_loguniform("l1_ratio", 0.01, 0.99))
    
    
    FOLD_NUM = 4
    kf = KFold(n_splits=FOLD_NUM,
              #shuffle=True,
              random_state=42)
    scores = []
    #feature_importance_df = pd.DataFrame()

    pred_cv = np.zeros(len(test.index))
    num_round = 10000

    


    for i, (tdx, vdx) in enumerate(kf.split(train_X[selected], train_y)):
        print(f'Fold : {i}')
        X_train, X_valid, y_train, y_valid = train_X[selected].iloc[tdx], train_X[selected].iloc[vdx], train_y.values[tdx], train_y.values[vdx]
        model = linear_model.ElasticNet(
            alpha=alpha, 
            l1_ratio=l1_ratio
            )
        model.fit(X_train, y_train)
        va_pred = model.predict(X_valid)
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

