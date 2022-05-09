# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

number_of_sub_datasets = 30  # サブデータセットの数
rate_of_selected_x_variables = 0.75  # 各サブデータセットで選択される説明変数の数の割合。0 より大きく 1 未満
fold_number = 10  # N-fold CV の N

svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # C の候補
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # ε の候補
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # γ の候補

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
autoscaled_y = (y - y.mean()) / y.std(ddof=1)

number_of_x_variables = int(np.ceil(x.shape[1] * rate_of_selected_x_variables))
print('各サブデータセットにおける説明変数の数 :', number_of_x_variables)
estimated_y_train_all = pd.DataFrame()  # 空の DataFrame 型の変数を作成し、ここにサブデータセットごとの y の推定結果を追加
selected_x_variable_numbers = []  # 空の list の変数を作成し、ここに各サブデータセットの説明変数の番号を追加
submodels = []  # 空の list の変数を作成し、ここに構築済みの各サブモデルを追加
for submodel_number in range(number_of_sub_datasets):
    print(submodel_number + 1, '/', number_of_sub_datasets)  # 進捗状況の表示
    # 説明変数の選択
    # 0 から 1 までの間に一様に分布する乱数を説明変数の数だけ生成して、その乱数値が小さい順に説明変数を選択
    random_x_variables = np.random.rand(x.shape[1])
    selected_x_variable_numbers_tmp = random_x_variables.argsort()[:number_of_x_variables]
    selected_autoscaled_x = autoscaled_x.iloc[:, selected_x_variable_numbers_tmp]
    selected_x_variable_numbers.append(selected_x_variable_numbers_tmp)

    # ハイパーパラメータの最適化
    # 分散最大化によるガウシアンカーネルのγの最適化
    variance_of_gram_matrix = []
    selected_autoscaled_x_array = np.array(selected_autoscaled_x)
    for nonlinear_svr_gamma in svr_gammas:
        gram_matrix = np.exp(- nonlinear_svr_gamma * ((selected_autoscaled_x_array[:, np.newaxis] - selected_autoscaled_x_array) ** 2).sum(axis=2))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_svr_gamma = svr_gammas[np.where(variance_of_gram_matrix==np.max(variance_of_gram_matrix))[0][0]]
    cross_validation = KFold(n_splits=fold_number, shuffle=True) # クロスバリデーションの分割の設定
    # CV による ε の最適化
    r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
    for svr_epsilon in svr_epsilons:
        model = SVR(kernel='rbf', C=3, epsilon=svr_epsilon, gamma=optimal_svr_gamma)
        autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)
        r2cvs.append(r2_score(y, autoscaled_estimated_y_in_cv * y.std() + y.mean()))
    optimal_svr_epsilon = svr_epsilons[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補
    
    # CV による C の最適化
    r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
    for svr_c in svr_cs:
        model = SVR(kernel='rbf', C=svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)
        autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)
        r2cvs.append(r2_score(y, autoscaled_estimated_y_in_cv * y.std() + y.mean()))
    optimal_svr_c = svr_cs[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補
    
    # CV による γ の最適化
    r2cvs = [] # 空の list。候補ごとに、クロスバリデーション後の r2 を入れていきます
    for svr_gamma in svr_gammas:
        model = SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=svr_gamma)
        autoscaled_estimated_y_in_cv = cross_val_predict(model, autoscaled_x, autoscaled_y, cv=cross_validation)
        r2cvs.append(r2_score(y, autoscaled_estimated_y_in_cv * y.std() + y.mean()))
    optimal_svr_gamma = svr_gammas[np.where(r2cvs==np.max(r2cvs))[0][0]] # クロスバリデーション後の r2 が最も大きい候補

    # SVR
    submodel = SVR(kernel='rbf', C=optimal_svr_c, epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma)  # モデルの宣言
    submodel.fit(selected_autoscaled_x, autoscaled_y)  # モデルの構築
    submodels.append(submodel)

# サブデータセットの説明変数の種類やサブデータセットを用いて構築されたモデルを保存。同じ名前のファイルがあるときは上書きされるため注意
pd.to_pickle(selected_x_variable_numbers, 'selected_x_variable_numbers_svr_gaussian.bin')
pd.to_pickle(submodels, 'submodels_svr_gaussian.bin')

# サブデータセットの説明変数の種類やサブデータセットを用いて構築されたモデルを読み込み
# 今回は、保存した後にすぐ読み込んでいるため、あまり意味はありませんが、サブデータセットの説明変数の種類や
# 構築されたモデルを保存しておくことで、後で新しいサンプルを予測したいときにモデル構築の過程を省略できます
selected_x_variable_numbers = pd.read_pickle('selected_x_variable_numbers_svr_gaussian.bin')
submodels = pd.read_pickle('submodels_svr_gaussian.bin')

# 予測用データセットの y の推定
estimated_y_prediction_all = pd.DataFrame()  # 空の DataFrame 型を作成し、ここにサブモデルごとの予測用データセットの y の推定結果を追加
for submodel_number in range(number_of_sub_datasets):
    # 説明変数の選択
    selected_autoscaled_x_prediction = autoscaled_x_prediction.iloc[:, selected_x_variable_numbers[submodel_number]]
    # 予測用データセットの y の推定
    estimated_y_prediction = pd.DataFrame(
        submodels[submodel_number].predict(selected_autoscaled_x_prediction))  # 予測用データセットの y の値を推定し、Pandas の DataFrame 型に変換
    estimated_y_prediction = estimated_y_prediction * y.std() + y.mean()  # スケールをもとに戻します
    estimated_y_prediction_all = pd.concat([estimated_y_prediction_all, estimated_y_prediction], axis=1)

# 予測用データセットの推定値の平均値
estimated_y_prediction = pd.DataFrame(estimated_y_prediction_all.mean(axis=1))  # Series 型のため、行名と列名の設定は別に
estimated_y_prediction.index = x_prediction.index
estimated_y_prediction.columns = ['estimated_y']
estimated_y_prediction.to_csv('estimated_y_prediction_ensemble_svr_gaussian.csv')  # 推定値を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

# 予測用データセットの推定値の標準偏差
std_of_estimated_y_prediction = pd.DataFrame(estimated_y_prediction_all.std(axis=1))  # Series 型のため、行名と列名の設定は別に
std_of_estimated_y_prediction.index = x_prediction.index
std_of_estimated_y_prediction.columns = ['std_of_estimated_y']
std_of_estimated_y_prediction.to_csv('std_of_estimated_y_prediction_ensemble_svr_gaussian.csv')  # 推定値の標準偏差を csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
