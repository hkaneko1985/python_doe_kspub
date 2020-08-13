# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor # RF モデルの構築に使用
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

number_of_test_samples = 5  # テストデータのサンプル数
x_variables_rates = np.arange(1, 11, dtype=float) / 10  # 決定木における X の数の割合
number_of_trees = 300  # サブデータセットの数

dataset = pd.read_csv('resin.csv', index_col=0, header=0)

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

# ランダムにトレーニングデータとテストデータとに分割
# random_state に数字を与えることで、別のときに同じ数字を使えば、ランダムとはいえ同じ結果にすることができます
if number_of_test_samples == 0:
    x_train = x.copy()
    x_test = x.copy()
    y_train = y.copy()
    y_test = y.copy()
else:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=number_of_test_samples, shuffle=True,
                                                        random_state=99)

# 標準偏差が 0 の特徴量の削除
deleting_variables = x_train.columns[x_train.std() == 0]
x_train = x_train.drop(deleting_variables, axis=1)
x_test = x_test.drop(deleting_variables, axis=1)

# OOB を用いた X の数の割合の最適化
r2_oob = [] # 空の list。説明変数の数の割合ごとに、OOB (Out Of Bag) の r2 を入れていきます
for x_variables_rate in x_variables_rates:
    model = RandomForestRegressor(n_estimators=number_of_trees,
                                  max_features=int(math.ceil(x_train.shape[1] * x_variables_rate)),
                                  oob_score=True)
    model.fit(x_train, y_train)
    r2_oob.append(r2_score(y_train, model.oob_prediction_))
# 結果の確認
plt.rcParams['font.size'] = 18
plt.scatter(x_variables_rates, r2_oob, c='blue')
plt.xlabel('rate of x-variables')
plt.ylabel('r2 in OOB')
plt.show()
optimal_x_variables_rate = x_variables_rates[np.where(r2_oob == np.max(r2_oob))[0][0]] # r2oob_allが最も大きい X の割合
print('最適化された決定木ごとの X の数 :', int(math.ceil(x_train.shape[1] * optimal_x_variables_rate)))

# モデル構築
model = RandomForestRegressor(n_estimators=number_of_trees,
                              max_features=int(math.ceil(x_train.shape[1] * optimal_x_variables_rate)),
                              oob_score=True) # RF モデルの宣言
model.fit(x_train, y_train)  # モデル構築

# 特徴量の重要度
variable_importances = pd.DataFrame(model.feature_importances_, index=x_train.columns, columns=['importances'])  # Pandas の DataFrame 型に変換
variable_importances.to_csv(
    'variable_importances_rf.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# トレーニングデータの推定
estimated_y_train = model.predict(x_train)  # y の推定
estimated_y_train = pd.DataFrame(estimated_y_train, index=x_train.index, columns=['estimated_y'])

# トレーニングデータの実測値 vs. 推定値のプロット
plt.rcParams['font.size'] = 18
plt.scatter(y_train, estimated_y_train.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
y_max = max(y_train.max(), estimated_y_train.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
y_min = min(y_train.min(), estimated_y_train.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
plt.xlabel('actual y')  # x 軸の名前
plt.ylabel('estimated y')  # y 軸の名前
plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
plt.show()  # 以上の設定で描画

# トレーニングデータのr2, RMSE, MAE
print('r^2 for training data :', r2_score(y_train, estimated_y_train))
print('RMSE for training data :', mean_squared_error(y_train, estimated_y_train, squared=False))
print('MAE for training data :', mean_absolute_error(y_train, estimated_y_train))

# トレーニングデータの結果の保存
y_train_for_save = pd.DataFrame(y_train)
y_train_for_save.columns = ['actual_y']
y_error_train = y_train_for_save.iloc[:, 0] - estimated_y_train.iloc[:, 0]
y_error_train = pd.DataFrame(y_error_train)
y_error_train.columns = ['error_of_y(actual_y-estimated_y)']
results_train = pd.concat([y_train_for_save, estimated_y_train, y_error_train], axis=1) # 結合
results_train.to_csv('estimated_y_train_in_detail_rf.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# テストデータの推定
estimated_y_test = model.predict(x_test)  # y の推定
estimated_y_test = pd.DataFrame(estimated_y_test, index=x_test.index, columns=['estimated_y'])

# テストデータの実測値 vs. 推定値のプロット
plt.rcParams['font.size'] = 18
plt.scatter(y_test, estimated_y_test.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
y_max = max(y_test.max(), estimated_y_test.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
y_min = min(y_test.min(), estimated_y_test.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
plt.xlabel('actual y')  # x 軸の名前
plt.ylabel('estimated y')  # y 軸の名前
plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
plt.show()  # 以上の設定で描画

# テストデータのr2, RMSE, MAE
print('r^2 for test data :', r2_score(y_test, estimated_y_test))
print('RMSE for test data :', mean_squared_error(y_test, estimated_y_test, squared=False))
print('MAE for test data :', mean_absolute_error(y_test, estimated_y_test))

# テストデータの結果の保存
y_test_for_save = pd.DataFrame(y_test)
y_test_for_save.columns = ['actual_y']
y_error_test = y_test_for_save.iloc[:, 0] - estimated_y_test.iloc[:, 0]
y_error_test = pd.DataFrame(y_error_test)
y_error_test.columns = ['error_of_y(actual_y-estimated_y)']
results_test = pd.concat([y_test_for_save, estimated_y_test, y_error_test], axis=1) # 結合
results_test.to_csv('estimated_y_test_in_detail_rf.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
