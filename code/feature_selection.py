import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

import lightgbm  as lgb

import copy


""" cross validation"""
def cross_validator(
    train_X: pd.DataFrame,
    train_y: pd.DataFrame
) -> float:
    scores = []

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        "seed":42,
        "learning_rate": 0.015469303108409655,
        "lambda_l1": 6.27078155451912e-08,
        "lambda_l2": 3.95710271939442e-08,
        "num_leaves": 206,
        "feature_fraction": 0.6203967979346209,
        "bagging_fraction": 0.8034268424574545,
        "bagging_freq": 2,
        "min_child_samples": 6
    }
    
    kf = KFold(n_splits=4, shuffle=True, random_state=71)
    for tr_idx, va_idx in kf.split(train_X):
        tr_X, va_X = train_X.iloc[tr_idx], train_X.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        
        tr_set = lgb.Dataset(tr_X, tr_y)
        va_set = lgb.Dataset(va_X, va_y, reference=tr_set)

        reg = lgb.train(params, tr_set, num_boost_round=2000, early_stopping_rounds=100,
                        valid_sets=[tr_set, va_set], verbose_eval=500)
        """reg = LGBMRegressor(class_weight=None,
        importance_type='split', learning_rate=0.1,
        n_iter_no_change=None, n_jobs=1,
        objective=None, param_distributions=None, random_state=71,
        study=None, timeout=None)
        reg.fit(tr_X, tr_y)"""
        va_pred = np.expm1(reg.predict(va_X))
        va_pred = np.where(va_pred < 0, 0, va_pred)
        #print(len(va_pred))
        #print(len(va_y))
        score = np.sqrt(mean_squared_log_error(np.expm1(va_y), va_pred))
        """score = np.sqrt(mean_squared_log_error(np.exp(va_y), va_pred))"""
        scores.append(score)
    return np.mean(scores)

def cross_validator_time(
    train: pd.DataFrame
) -> float:
    scores = []

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        "seed":42,
        "learning_rate": 0.015469303108409655,
        "lambda_l1": 6.27078155451912e-08,
        "lambda_l2": 3.95710271939442e-08,
        "num_leaves": 206,
        "feature_fraction": 0.6203967979346209,
        "bagging_fraction": 0.8034268424574545,
        "bagging_freq": 2,
        "min_child_samples": 6
    }
    
    kf = KFold(n_splits=4, shuffle=True, random_state=71)
    va_periods = [20164, 20153, 20142, 20131]
    for va_period in va_periods:
        print(f'Fold : {va_period}')
        
        print(len(train))
        X_train, X_valid, y_train, y_valid = train[train["Period"] < va_period], train[train["Period"] >= va_period], np.log1p(train["y"][train["Period"] < va_period]), np.log1p(train["y"][train["Period"] >= va_period])
        
        tr_set = lgb.Dataset(X_train, y_train)
        va_set = lgb.Dataset(X_valid, y_valid, reference=tr_set)

        reg = lgb.train(params, tr_set, num_boost_round=2000, early_stopping_rounds=100,
                        valid_sets=[tr_set, va_set], verbose_eval=500)
        
        va_pred = np.expm1(reg.predict(X_valid))
        va_pred = np.where(va_pred < 0, 0, va_pred)
        #print(len(va_pred))
        #print(len(va_y))
        score = np.sqrt(mean_squared_log_error(np.expm1(y_valid), va_pred))
        """score = np.sqrt(mean_squared_log_error(np.exp(va_y), va_pred))"""
        scores.append(score)
        train = train[train["Period"] < va_period]
    return np.mean(scores)

def greedy_forward_selection_time(
        train: pd.DataFrame
    ) -> set:
        

        best_score = 99999.0
        candidates = np.random.RandomState(71).permutation(train.drop(["y"], axis=1).columns)
        selected = set([])

        print("start simple selection")
        for feature in candidates:
            fs = list(selected) + [feature]
            score = cross_validator_time(train)
            
            if score < best_score:
                selected.add(feature)
                best_score = score
                #print(f'selected: {feature}')
                #print(f'score: {score}')

        print(f'selected features: {selected}')

        return selected

class FeatureSelector:
    def __init__(
        self,
        train_X: pd.DataFrame,
        train_y: pd.DataFrame
    ) -> None:
        self.train_X = train_X
        self.train_y = train_y


    """ Greedy Forward Selection"""
    def greedy_forward_selection(
        self
    ) -> set:
        

        best_score = 99999.0
        candidates = np.random.RandomState(71).permutation(self.train_X.columns)
        selected = set([])

        print("start simple selection")
        for feature in candidates:
            fs = list(selected) + [feature]
            score = cross_validator(self.train_X[fs], self.train_y)
            
            
            if score < best_score:
                selected.add(feature)
                best_score = score
                #print(f'selected: {feature}')
                #print(f'score: {score}')

        print(f'selected features: {selected}')

        return selected


    """ target encoding"""
    def target_encoder(
        self,
        test: pd.DataFrame
        ) -> pd.DataFrame:
        train_X_te = copy.deepcopy(self.train_X)
        test_X_te = copy.deepcopy(test)
        column_list = self.train_X.select_dtypes(include=["category"]).columns
        if len(column_list) > 0:
            print("categorical features:"+ str(column_list))
        else:
            print("No categorical features")
        for c in column_list:
            data_tmp = pd.DataFrame({c: train_X_te[c], "target": self.train_y})
            target_mean = data_tmp.groupby(c)["target"].mean()
            """ print(target_mean)"""
            test_X_te[c] = test_X_te[c].map(target_mean).astype(float)

            tmp = np.repeat(np.nan, self.train_X.shape[0])
            kf_encoding = KFold(n_splits=4, shuffle=True, random_state=72)
            for idx_1, idx_2 in kf_encoding.split(train_X_te):
                target_mean = data_tmp.iloc[idx_1].groupby(c)["target"].mean()

                tmp[idx_2] = train_X_te[c].iloc[idx_2].map(target_mean)

            train_X_te[c] = tmp

        return train_X_te, test_X_te