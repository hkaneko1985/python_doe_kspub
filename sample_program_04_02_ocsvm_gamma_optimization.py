# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

ocsvm_nu = 0.04  # OCSVM における ν。トレーニングデータにおけるサンプル数に対する、サポートベクターの数の下限の割合
ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)  # OCSVM における γ の候補

dataset = pd.read_csv('resin.csv', index_col=0, header=0)
x_prediction = pd.read_csv('resin_prediction.csv', index_col=0, header=0)

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

# 標準偏差が 0 の特徴量の削除
deleting_variables = x.columns[x.std() == 0]
x = x.drop(deleting_variables, axis=1)
x_prediction = x_prediction.drop(deleting_variables, axis=1)

# オートスケーリング
autoscaled_x = (x - x.mean()) / x.std()
autoscaled_x_prediction = (x_prediction - x.mean()) / x.std()

# 分散最大化によるガウシアンカーネルのγの最適化
variance_of_gram_matrix = []
autoscaled_x_array = np.array(autoscaled_x)
for nonlinear_svr_gamma in ocsvm_gammas:
    gram_matrix = np.exp(- nonlinear_svr_gamma * ((autoscaled_x_array[:, np.newaxis] - autoscaled_x_array) ** 2).sum(axis=2))
    variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
optimal_gamma = ocsvm_gammas[np.where(variance_of_gram_matrix==np.max(variance_of_gram_matrix))[0][0]]
# 最適化された γ
print('最適化された gamma :', optimal_gamma)

# OCSVM による AD
ad_model = OneClassSVM(kernel='rbf', gamma=optimal_gamma, nu=ocsvm_nu)  # AD モデルの宣言
ad_model.fit(autoscaled_x)  # モデル構築

# トレーニングデータのデータ密度 (f(x) の値)
data_density_train = ad_model.decision_function(autoscaled_x)
number_of_support_vectors = len(ad_model.support_)
number_of_outliers_in_training_data = sum(data_density_train < 0)
print('\nトレーニングデータにおけるサポートベクター数 :', number_of_support_vectors)
print('トレーニングデータにおけるサポートベクターの割合 :', number_of_support_vectors / x.shape[0])
print('\nトレーニングデータにおける外れサンプル数 :', number_of_outliers_in_training_data)
print('トレーニングデータにおける外れサンプルの割合 :', number_of_outliers_in_training_data / x.shape[0])
data_density_train = pd.DataFrame(data_density_train, index=x.index, columns=['ocsvm_data_density'])
data_density_train.to_csv('ocsvm_gamma_optimization_data_density_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# トレーニングデータに対して、AD の中か外かを判定
inside_ad_flag_train = data_density_train >= 0
inside_ad_flag_train.columns = ['inside_ad_flag']
inside_ad_flag_train.to_csv('inside_ad_flag_train_ocsvm_gamma_optimization.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# 予測用データセットのデータ密度 (f(x) の値)
data_density_prediction = ad_model.decision_function(autoscaled_x_prediction)
number_of_outliers_in_prediction_data = sum(data_density_prediction < 0)
print('\n予測用データセットにおける外れサンプル数 :', number_of_outliers_in_prediction_data)
print('予測用データセットにおける外れサンプルの割合 :', number_of_outliers_in_prediction_data / x_prediction.shape[0])
data_density_prediction = pd.DataFrame(data_density_prediction, index=x_prediction.index, columns=['ocsvm_data_density'])
data_density_prediction.to_csv('ocsvm_gamma_optimization_data_density_prediction.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# 予測用データセットに対して、AD の中か外かを判定
inside_ad_flag_prediction = data_density_prediction >= 0
inside_ad_flag_prediction.columns = ['inside_ad_flag']
inside_ad_flag_prediction.to_csv('inside_ad_flag_prediction_ocsvm_gamma_optimization.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
