# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, OneClassSVM
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors

regression_method = 'gpr_one_kernel'  # 回帰分析手法 'ols_linear', 'ols_nonlinear', 'svr_linear', 'svr_gaussian', 'gpr_one_kernel', 'gpr_kernels'
ad_method = 'ocsvm'  # AD設定手法 'knn', 'ocsvm', 'ocsvm_gamma_optimization'

fold_number = 10  # クロスバリデーションの fold 数
rate_of_training_samples_inside_ad = 0.96  # AD 内となるトレーニングデータの割合。AD　のしきい値を決めるときに使用

linear_svr_cs = 2 ** np.arange(-10, 5, dtype=float) # 線形SVR の C の候補
linear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float) # 線形SVRの ε の候補
nonlinear_svr_cs = 2 ** np.arange(-5, 10, dtype=float) # SVR の C の候補
nonlinear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float) # SVR の ε の候補
nonlinear_svr_gammas = 2 ** np.arange(-20, 10, dtype=float) # SVR のガウシアンカーネルの γ の候補
kernel_number = 2  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
k_in_knn = 5  # k-NN における k
ocsvm_nu = 0.04  # OCSVM における ν。トレーニングデータにおけるサンプル数に対する、サポートベクターの数の下限の割合
ocsvm_gamma = 0.1  # OCSVM における γ
ocsvm_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

dataset = pd.read_csv('resin.csv', index_col=0, header=0)
x_prediction = pd.read_csv('remaining_samples.csv', index_col=0, header=0)

# データ分割
y = dataset.iloc[:, 0]  # 目的変数
x = dataset.iloc[:, 1:]  # 説明変数

# 非線形変換
if regression_method == 'ols_nonlinear':
    x_tmp = x.copy()
    x_prediction_tmp = x_prediction.copy()
    x_square = x ** 2  # 二乗項
    x_prediction_square = x_prediction ** 2  # 二乗項
    # 追加
    print('\n二乗項と交差項の追加')
    for i in range(x_tmp.shape[1]):
        print(i + 1, '/', x_tmp.shape[1])
        for j in range(x_tmp.shape[1]):
            if i == j:  # 二乗項
                x = pd.concat([x, x_square.rename(columns={x_square.columns[i]: '{0}^2'.format(x_square.columns[i])}).iloc[:, i]], axis=1)
                x_prediction = pd.concat([x_prediction, x_prediction_square.rename(columns={x_prediction_square.columns[i]: '{0}^2'.format(x_prediction_square.columns[i])}).iloc[:, i]], axis=1)
            elif i < j:  # 交差項
                x_cross = x_tmp.iloc[:, i] * x_tmp.iloc[:, j]
                x_prediction_cross = x_prediction_tmp.iloc[:, i] * x_prediction_tmp.iloc[:, j]
                x_cross.name = '{0}*{1}'.format(x_tmp.columns[i], x_tmp.columns[j])
                x_prediction_cross.name = '{0}*{1}'.format(x_prediction_tmp.columns[i], x_prediction_tmp.columns[j])
                x = pd.concat([x, x_cross], axis=1)
                x_prediction = pd.concat([x_prediction, x_prediction_cross], axis=1)

# 標準偏差が 0 の特徴量の削除
deleting_variables = x.columns[x.std() == 0]
x = x.drop(deleting_variables, axis=1)
x_prediction = x_prediction.drop(deleting_variables, axis=1)

# カーネル 11 種類
kernels = [ConstantKernel() * DotProduct() + WhiteKernel(),
           ConstantKernel() * RBF() + WhiteKernel(),
           ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
           ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel(),
           ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
           ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
           ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
           ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
           ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
           ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
           ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()]

# オートスケーリング
autoscaled_y = (y - y.mean()) / y.std()
autoscaled_x = (x - x.mean()) / x.std()
x_prediction.columns = x.columns
autoscaled_x_prediction = (x_prediction - x.mean()) / x.std()

# モデル構築
if regression_method == 'ols_linear' or regression_method == 'ols_nonlinear':
    model = LinearRegression()
elif regression_method == 'svr_linear':
    # クロスバリデーションによる C, ε の最適化
    cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
    gs_cv = GridSearchCV(SVR(kernel='linear'), {'C':linear_svr_cs, 'epsilon':linear_svr_epsilons}, cv=cross_validation)  # グリッドサーチの設定
    gs_cv.fit(autoscaled_x, autoscaled_y)  # グリッドサーチ + クロスバリデーション実施
    optimal_linear_svr_c = gs_cv.best_params_['C']  # 最適な C
    optimal_linear_svr_epsilon = gs_cv.best_params_['epsilon']  # 最適な ε
    print('最適化された C : {0} (log(C)={1})'.format(optimal_linear_svr_c, np.log2(optimal_linear_svr_c)))
    print('最適化された ε : {0} (log(ε)={1})'.format(optimal_linear_svr_epsilon, np.log2(optimal_linear_svr_epsilon)))
    model = SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon) # SVRモデルの宣言
elif regression_method == 'svr_gaussian':
    # C, ε, γの最適化
    # 分散最大化によるガウシアンカーネルのγの最適化
    variance_of_gram_matrix = []
    autoscaled_x_array = np.array(autoscaled_x)
    for nonlinear_svr_gamma in nonlinear_svr_gammas:
        gram_matrix = np.exp(- nonlinear_svr_gamma * ((autoscaled_x_array[:, np.newaxis] - autoscaled_x_array) ** 2).sum(axis=2))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_nonlinear_gamma = nonlinear_svr_gammas[np.where(variance_of_gram_matrix==np.max(variance_of_gram_matrix))[0][0]]
    
    cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
    # CV による ε の最適化
    r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
    for nonlinear_svr_epsilon in nonlinear_svr_epsilons:
        model = SVR(kernel='rbf', C=3, epsilon=nonlinear_svr_epsilon, gamma=optimal_nonlinear_gamma)
        autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)
        r2cvs.append(r2_score(y, autoscaled_estimated_y_in_cv * y.std() + y.mean()))
    optimal_nonlinear_epsilon = nonlinear_svr_epsilons[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補
    
    # CV による C の最適化
    r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
    for nonlinear_svr_c in nonlinear_svr_cs:
        model = SVR(kernel='rbf', C=nonlinear_svr_c, epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma)
        autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)
        r2cvs.append(r2_score(y, autoscaled_estimated_y_in_cv * y.std() + y.mean()))
    optimal_nonlinear_c = nonlinear_svr_cs[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補
    
    # CV による γ の最適化
    r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
    for nonlinear_svr_gamma in nonlinear_svr_gammas:
        model = SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon, gamma=nonlinear_svr_gamma)
        autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)
        r2cvs.append(r2_score(y, autoscaled_estimated_y_in_cv * y.std() + y.mean()))
    optimal_nonlinear_gamma = nonlinear_svr_gammas[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補
    # 結果の確認
    print('最適化された C : {0} (log(C)={1})'.format(optimal_nonlinear_c, np.log2(optimal_nonlinear_c)))
    print('最適化された ε : {0} (log(ε)={1})'.format(optimal_nonlinear_epsilon, np.log2(optimal_nonlinear_epsilon)))
    print('最適化された γ : {0} (log(γ)={1})'.format(optimal_nonlinear_gamma, np.log2(optimal_nonlinear_gamma)))
    # モデル構築
    model = SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma)  # SVR モデルの宣言
elif regression_method == 'gpr_one_kernel':
    selected_kernel = kernels[kernel_number]
    model = GaussianProcessRegressor(alpha=0, kernel=selected_kernel)
elif regression_method == 'gpr_kernels':
    # クロスバリデーションによるカーネル関数の最適化
    cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
    r2cvs = [] # 空の list。主成分の数ごとに、クロスバリデーション後の r2 を入れていきます
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

# 標準回帰係数
if regression_method == 'ols_linear' or regression_method == 'ols_nonlinear' or regression_method == 'svr_linear':
    if regression_method == 'svr_linear':
        standard_regression_coefficients = model.coef_.T
    else:
        standard_regression_coefficients = model.coef_
    standard_regression_coefficients = pd.DataFrame(standard_regression_coefficients, index=x.columns, columns=['standard_regression_coefficients'])
    standard_regression_coefficients.to_csv(
        'standard_regression_coefficients_{0}.csv'.format(regression_method))  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# トレーニングデータの推定
autoscaled_estimated_y = model.predict(autoscaled_x)  # y の推定
estimated_y = autoscaled_estimated_y * y.std() + y.mean()  # スケールをもとに戻す
estimated_y = pd.DataFrame(estimated_y, index=x.index, columns=['estimated_y'])

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
results_train = pd.concat([y_for_save, estimated_y, y_error_train], axis=1) # 結合
results_train.to_csv('estimated_y_in_detail_{0}.csv'.format(regression_method))  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# クロスバリデーションによる y の値の推定
cross_validation = KFold(n_splits=fold_number, random_state=9, shuffle=True) # クロスバリデーションの分割の設定
autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)  # y の推定
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
if regression_method == 'gpr_one_kernel' or regression_method == 'gpr_kernels':  # 標準偏差あり
    estimated_y_prediction, estimated_y_prediction_std = model.predict(autoscaled_x_prediction, return_std=True)
    estimated_y_prediction_std = estimated_y_prediction_std * y.std()
    estimated_y_prediction_std = pd.DataFrame(estimated_y_prediction_std, x_prediction.index, columns=['std_of_estimated_y'])
    estimated_y_prediction_std.to_csv('estimated_y_prediction_{0}_std.csv'.format(regression_method))  # 予測値の標準偏差を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
else:
    estimated_y_prediction = model.predict(autoscaled_x_prediction)

estimated_y_prediction = estimated_y_prediction * y.std() + y.mean()
estimated_y_prediction = pd.DataFrame(estimated_y_prediction, x_prediction.index, columns=['estimated_y'])
estimated_y_prediction.to_csv('estimated_y_prediction_{0}.csv'.format(regression_method))  # 予測結果を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# 非線形変換を戻す
if regression_method == 'ols_nonlinear':
    x = x_tmp.copy()
    x_prediction = x_prediction_tmp.copy()
    # 標準偏差が 0 の特徴量の削除
    deleting_variables = x.columns[x.std() == 0]
    x = x.drop(deleting_variables, axis=1)
    x_prediction = x_prediction.drop(deleting_variables, axis=1)    
    # オートスケーリング
    autoscaled_x = (x - x.mean()) / x.std()
    autoscaled_x_prediction = (x_prediction - x.mean()) / x.std()

# AD
if ad_method == 'knn':
    ad_model = NearestNeighbors(n_neighbors=k_in_knn, metric='euclidean')
    ad_model.fit(autoscaled_x)
    
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
    inside_ad_flag_train = mean_of_knn_distance_train <= ad_threshold
    
    # 予測用データに対する k-NN 距離の計算
    knn_distance_prediction, knn_index_prediction = ad_model.kneighbors(autoscaled_x_prediction)
    knn_distance_prediction = pd.DataFrame(knn_distance_prediction, index=x_prediction.index)  # DataFrame型に変換
    ad_index_prediction = pd.DataFrame(knn_distance_prediction.mean(axis=1), columns=['mean_of_knn_distance'])  # k_in_knn 個の距離の平均
    inside_ad_flag_prediction = ad_index_prediction <= ad_threshold

elif ad_method == 'ocsvm':
    if ad_method == 'ocsvm_gamma_optimization':
        # 分散最大化によるガウシアンカーネルのγの最適化
        variance_of_gram_matrix = []
        autoscaled_x_array = np.array(autoscaled_x)
        for nonlinear_svr_gamma in ocsvm_gammas:
            gram_matrix = np.exp(- nonlinear_svr_gamma * ((autoscaled_x_array[:, np.newaxis] - autoscaled_x_array) ** 2).sum(axis=2))
            variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
        optimal_gamma = ocsvm_gammas[np.where(variance_of_gram_matrix==np.max(variance_of_gram_matrix))[0][0]]
        # 最適化された γ
        print('最適化された gamma :', optimal_gamma)
    else:
        optimal_gamma = ocsvm_gamma
    
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
    data_density_train.to_csv('ocsvm_data_density_train.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
    # トレーニングデータに対して、AD の中か外かを判定
    inside_ad_flag_train = data_density_train >= 0
    # 予測用データのデータ密度 (f(x) の値)
    ad_index_prediction = ad_model.decision_function(autoscaled_x_prediction)
    number_of_outliers_in_prediction_data = sum(ad_index_prediction < 0)
    print('\nテストデータにおける外れサンプル数 :', number_of_outliers_in_prediction_data)
    print('テストデータにおける外れサンプルの割合 :', number_of_outliers_in_prediction_data / x_prediction.shape[0])
    ad_index_prediction = pd.DataFrame(ad_index_prediction, index=x_prediction.index, columns=['ocsvm_data_density'])
    ad_index_prediction.to_csv('ocsvm_ad_index_prediction.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
    # 予測用トデータに対して、AD の中か外かを判定
    inside_ad_flag_prediction = ad_index_prediction >= 0

estimated_y_prediction[np.logical_not(inside_ad_flag_prediction)] = -10 ** 10 # AD 外の候補においては負に非常に大きい値を代入し、次の候補として選ばれないようにします

# 保存
inside_ad_flag_train.columns = ['inside_ad_flag']
inside_ad_flag_train.to_csv('inside_ad_flag_train_{0}.csv'.format(ad_method))  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
inside_ad_flag_prediction.columns = ['inside_ad_flag']
inside_ad_flag_prediction.to_csv('inside_ad_flag_prediction_{0}.csv'.format(ad_method))  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
ad_index_prediction.to_csv('ad_index_prediction_{0}.csv'.format(ad_method))  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
estimated_y_prediction.to_csv('estimated_y_prediction_considering_ad_{0}_{1}.csv'.format(regression_method, ad_method)) # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# 次のサンプル
next_sample = x_prediction.iloc[estimated_y_prediction.idxmax(), :]  # 次のサンプル
next_sample.to_csv('next_sample_{0}_{1}.csv'.format(regression_method, ad_method)) # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
