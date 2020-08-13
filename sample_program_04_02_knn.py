# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import pandas as pd
from sklearn.neighbors import NearestNeighbors  # k-NN

k_in_knn = 5  # k-NN における k
rate_of_training_samples_inside_ad = 0.96  # AD 内となるトレーニングデータの割合。AD　のしきい値を決めるときに使用

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

# k-NN による AD
ad_model = NearestNeighbors(n_neighbors=k_in_knn, metric='euclidean')  # AD モデルの宣言
ad_model.fit(autoscaled_x)  # k-NN による AD では、トレーニングデータの x を model_ad に格納することに対応

# サンプルごとの k 最近傍サンプルとの距離に加えて、k 最近傍サンプルのインデックス番号も一緒に出力されるため、出力用の変数を 2 つに
# トレーニングデータでは k 最近傍サンプルの中に自分も含まれ、自分との距離の 0 を除いた距離を考える必要があるため、k_in_knn + 1 個と設定
knn_distance_train, knn_index_train = ad_model.kneighbors(autoscaled_x, n_neighbors=k_in_knn + 1)
knn_distance_train = pd.DataFrame(knn_distance_train, index=autoscaled_x.index)  # DataFrame型に変換
mean_of_knn_distance_train = pd.DataFrame(knn_distance_train.iloc[:, 1:].mean(axis=1),
                                          columns=['mean_of_knn_distance'])  # 自分以外の k_in_knn 個の距離の平均
mean_of_knn_distance_train.to_csv('mean_of_knn_distance_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# トレーニングデータのサンプルの rate_of_training_samples_inside_ad * 100 % が含まれるようにしきい値を設定
sorted_mean_of_knn_distance_train = mean_of_knn_distance_train.iloc[:, 0].sort_values(ascending=True)  # 距離の平均の小さい順に並び替え
ad_threshold = sorted_mean_of_knn_distance_train.iloc[
    round(autoscaled_x.shape[0] * rate_of_training_samples_inside_ad) - 1]

# トレーニングデータに対して、AD の中か外かを判定
inside_ad_flag_train = mean_of_knn_distance_train <= ad_threshold   # AD 内のサンプルのみ TRUE
inside_ad_flag_train.columns=['inside_ad_flag']
inside_ad_flag_train.to_csv('inside_ad_flag_train_knn.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# 予測用データに対する k-NN 距離の計算
knn_distance_prediction, knn_index_prediction = ad_model.kneighbors(autoscaled_x_prediction)
knn_distance_prediction = pd.DataFrame(knn_distance_prediction, index=x_prediction.index)  # DataFrame型に変換
mean_of_knn_distance_prediction = pd.DataFrame(knn_distance_prediction.mean(axis=1),
                                         columns=['mean_of_knn_distance'])  # k_in_knn 個の距離の平均
mean_of_knn_distance_prediction.to_csv('mean_of_knn_distance_prediction.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

# 予測用データに対して、AD の中か外かを判定
inside_ad_flag_prediction = mean_of_knn_distance_prediction <= ad_threshold  # AD 内のサンプルのみ TRUE
inside_ad_flag_prediction.columns=['inside_ad_flag']
inside_ad_flag_prediction.to_csv('inside_ad_flag_prediction_knn.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
