# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

regression_method = 'gpr_one_kernel'  # gpr_one_kernel', 'gpr_kernels'
acquisition_function = 'PTR'  # 'PTR', 'PI', 'EI', 'MI'

fold_number = 10  # クロスバリデーションの fold 数
kernel_number = 2  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
target_range = [0, 1]  # PTR
relaxation = 0.01  # EI, PI
delta = 10 ** -6  # MI

dataset = pd.read_csv('resin.csv', index_col=0, header=0)
x_prediction = pd.read_csv('remaining_samples.csv', index_col=0, header=0)

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

# 標準偏差が 0 の特徴量の削除
deleting_variables = x.columns[x.std() == 0]
x = x.drop(deleting_variables, axis=1)
x_prediction = x_prediction.drop(deleting_variables, axis=1)

# カーネル 11 種類
kernels = [DotProduct() + WhiteKernel(),
           ConstantKernel() * RBF() + WhiteKernel(),
           ConstantKernel() * RBF() + WhiteKernel() + DotProduct(),
           ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel(),
           ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel() + DotProduct(),
           ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
           ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + DotProduct(),
           ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
           ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + DotProduct(),
           ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
           ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + DotProduct()]

# オートスケーリング
autoscaled_y = (y - y.mean()) / y.std()
autoscaled_x = (x - x.mean()) / x.std()
autoscaled_x_prediction = (x_prediction - x.mean()) / x.std()

# モデル構築
if regression_method == 'gpr_one_kernel':
    selected_kernel = kernels[kernel_number]
    model = GaussianProcessRegressor(alpha=0, kernel=selected_kernel)
elif regression_method == 'gpr_kernels':
    # クロスバリデーションによるカーネル関数の最適化
    cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
    r2cvs = [] # 空の list。カーネル関数ごとに、クロスバリデーション後の r2 を入れていきます
    for index, kernel in enumerate(kernels):
        print(index + 1, '/', len(kernels))
        model = GaussianProcessRegressor(alpha=0, kernel=kernel)
        estimated_y_in_cv = np.ndarray.flatten(cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation))
        estimated_y_in_cv = estimated_y_in_cv * y.std(ddof=1) + y.mean()
        r2cvs.append(r2_score(y, estimated_y_in_cv))
    optimal_kernel_number = np.where(r2cvs == np.max(r2cvs))[0][0]  # クロスバリデーション後の r2 が最も大きいカーネル関数の番号
    optimal_kernel = kernels[optimal_kernel_number]  # クロスバリデーション後の r2 が最も大きいカーネル関数
    print('クロスバリデーションで選択されたカーネル関数の番号 :', optimal_kernel_number)
    print('クロスバリデーションで選択されたカーネル関数 :', optimal_kernel)
    
    # モデル構築
    model = GaussianProcessRegressor(alpha=0, kernel=optimal_kernel) # GPR モデルの宣言
    
model.fit(autoscaled_x, autoscaled_y)  # モデル構築

# トレーニングデータの推定
autoscaled_estimated_y, autoscaled_estimated_y_std = model.predict(autoscaled_x, return_std=True)  # y の推定
estimated_y = autoscaled_estimated_y * y.std() + y.mean()  # スケールをもとに戻す
estimated_y_std = autoscaled_estimated_y_std * y.std()  # スケールをもとに戻す
estimated_y = pd.DataFrame(estimated_y, index=x.index, columns=['estimated_y'])
estimated_y_std = pd.DataFrame(estimated_y_std, index=x.index, columns=['std_of_estimated_y'])


# トレーニングデータの実測値 vs. 推定値のプロット
plt.rcParams['font.size'] = 18
plt.scatter(y, estimated_y.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
y_max = max(y.max(), estimated_y.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
y_min = min(y.min(), estimated_y.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
plt.xlabel('actual y')  # x 軸の名前
plt.ylabel('estimated y')  # y 軸の名前
plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
plt.show()  # 以上の設定で描画

# トレーニングデータのr2, RMSE, MAE
print('r^2 for training data :', r2_score(y, estimated_y))
print('RMSE for training data :', mean_squared_error(y, estimated_y, squared=False))
print('MAE for training data :', mean_absolute_error(y, estimated_y))

# トレーニングデータの結果の保存
y_for_save = pd.DataFrame(y)
y_for_save.columns = ['actual_y']
y_error_train = y_for_save.iloc[:, 0] - estimated_y.iloc[:, 0]
y_error_train = pd.DataFrame(y_error_train)
y_error_train.columns = ['error_of_y(actual_y-estimated_y)']
results_train = pd.concat([y_for_save, estimated_y, y_error_train, estimated_y_std], axis=1) # 結合
results_train.to_csv('estimated_y_in_detail_{0}.csv'.format(regression_method))  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# クロスバリデーションによる y の値の推定
cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y)  # y の推定
estimated_y_in_cv = autoscaled_estimated_y_in_cv * y.std() + y.mean()  # スケールをもとに戻す
estimated_y_in_cv = pd.DataFrame(estimated_y_in_cv, index=x.index, columns=['estimated_y'])

# クロスバリデーションにおける実測値 vs. 推定値のプロット
plt.rcParams['font.size'] = 18
plt.scatter(y, estimated_y_in_cv.iloc[:, 0], c='blue')  # 実測値 vs. 推定値プロット
y_max = max(y.max(), estimated_y_in_cv.iloc[:, 0].max())  # 実測値の最大値と、推定値の最大値の中で、より大きい値を取得
y_min = min(y.min(), estimated_y_in_cv.iloc[:, 0].min())  # 実測値の最小値と、推定値の最小値の中で、より小さい値を取得
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')  # 取得した最小値-5%から最大値+5%まで、対角線を作成
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # y 軸の範囲の設定
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))  # x 軸の範囲の設定
plt.xlabel('actual y')  # x 軸の名前
plt.ylabel('estimated y')  # y 軸の名前
plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
plt.show()  # 以上の設定で描画

# クロスバリデーションにおけるr2, RMSE, MAE
print('r^2 in cross-validation :', r2_score(y, estimated_y_in_cv))
print('RMSE in cross-validation :', mean_squared_error(y, estimated_y_in_cv, squared=False))
print('MAE in cross-validation :', mean_absolute_error(y, estimated_y_in_cv))

# クロスバリデーションの結果の保存
y_error_in_cv = y_for_save.iloc[:, 0] - estimated_y_in_cv.iloc[:, 0]
y_error_in_cv = pd.DataFrame(y_error_in_cv)
y_error_in_cv.columns = ['error_of_y(actual_y-estimated_y)']
results_in_cv = pd.concat([y_for_save, estimated_y_in_cv, y_error_in_cv], axis=1) # 結合
results_in_cv.to_csv('estimated_y_in_cv_in_detail_{0}.csv'.format(regression_method))  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# 予測
estimated_y_prediction, estimated_y_prediction_std = model.predict(autoscaled_x_prediction, return_std=True)
estimated_y_prediction = estimated_y_prediction * y.std() + y.mean()
estimated_y_prediction_std = estimated_y_prediction_std * y.std()

# 獲得関数の計算
cumulative_variance = np.zeros(x_prediction.shape[0]) # MI で必要な "ばらつき" を 0 で初期化
if acquisition_function == 'MI':
    acquisition_function_prediction = estimated_y_prediction + np.log(2 / delta) ** 0.5 * (
            (estimated_y_prediction_std ** 2 + cumulative_variance) ** 0.5 - cumulative_variance ** 0.5)
    cumulative_variance = cumulative_variance + estimated_y_prediction_std ** 2
elif acquisition_function == 'EI':
    acquisition_function_prediction = (estimated_y_prediction - max(y) - relaxation * y.std()) * \
                                       norm.cdf((estimated_y_prediction - max(y) - relaxation * y.std()) /
                                                 estimated_y_prediction_std) + \
                                       estimated_y_prediction_std * \
                                       norm.pdf((estimated_y_prediction - max(y) - relaxation * y.std()) /
                                                 estimated_y_prediction_std)
elif acquisition_function == 'PI':
    acquisition_function_prediction = norm.cdf(
            (estimated_y_prediction - max(y) - relaxation * y.std()) / estimated_y_prediction_std)
elif acquisition_function == 'PTR':
    acquisition_function_prediction = norm.cdf(target_range[1],
                                               loc=estimated_y_prediction,
                                               scale=estimated_y_prediction_std
                                               ) - norm.cdf(target_range[0],
                                                            loc=estimated_y_prediction,
                                                            scale=estimated_y_prediction_std)
acquisition_function_prediction[estimated_y_prediction_std <= 0] = 0

# 保存
estimated_y_prediction = pd.DataFrame(estimated_y_prediction, x_prediction.index, columns=['estimated_y'])
estimated_y_prediction.to_csv('estimated_y_prediction_{0}.csv'.format(regression_method))  # 予測結果を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
estimated_y_prediction_std = pd.DataFrame(estimated_y_prediction_std, x_prediction.index, columns=['std_of_estimated_y'])
estimated_y_prediction_std.to_csv('estimated_y_prediction_{0}_std.csv'.format(regression_method))  # 予測値の標準偏差を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
acquisition_function_prediction = pd.DataFrame(acquisition_function_prediction, index=x_prediction.index, columns=['acquisition_function'])
acquisition_function_prediction.to_csv('acquisition_function_prediction_{0}_{1}.csv'.format(regression_method, acquisition_function))  # 獲得関数を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# 次のサンプル
next_sample = x_prediction.iloc[acquisition_function_prediction.idxmax(), :]  # 次のサンプル
next_sample.to_csv('next_sample_bo_{0}_{1}.csv'.format(regression_method, acquisition_function)) # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
