# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import pandas as pd
import numpy as np

number_of_selecting_samples = 30  # 選択するサンプル数
number_of_random_searches = 1000  # ランダムにサンプルを選択して D 最適基準を計算する繰り返し回数

x_generated = pd.read_csv('generated_samples.csv', index_col=0, header=0)
autoscaled_x_generated = (x_generated - x_generated.mean()) / x_generated.std()

# 実験条件の候補のインデックスの作成
all_indexes = list(range(x_generated.shape[0]))

# D 最適基準に基づくサンプル選択
np.random.seed(11) # 乱数を生成するためのシードを固定
for random_search_number in range(number_of_random_searches):
    # 1. ランダムに候補を選択
    new_selected_indexes = np.random.choice(all_indexes, number_of_selecting_samples, replace=False)
    new_selected_samples = autoscaled_x_generated.iloc[new_selected_indexes, :]
    # 2. D 最適基準を計算
    xt_x = np.dot(new_selected_samples.T, new_selected_samples)
    d_optimal_value = np.linalg.det(xt_x) 
    # 3. D 最適基準が前回までの最大値を上回ったら、選択された候補を更新
    if random_search_number == 0:
        best_d_optimal_value = d_optimal_value.copy()
        selected_sample_indexes = new_selected_indexes.copy()
    else:
        if best_d_optimal_value < d_optimal_value:
            best_d_optimal_value = d_optimal_value.copy()
            selected_sample_indexes = new_selected_indexes.copy()
selected_sample_indexes = list(selected_sample_indexes) # リスト型に変換

# 選択されたサンプル、選択されなかったサンプル
selected_samples = x_generated.iloc[selected_sample_indexes, :]  # 選択されたサンプル
remaining_indexes = np.delete(all_indexes, selected_sample_indexes)  # 選択されなかったサンプルのインデックス
remaining_samples = x_generated.iloc[remaining_indexes, :]  # 選択されなかったサンプル

# 保存
selected_samples.to_csv('selected_samples.csv')  # 選択されたサンプルを csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
remaining_samples.to_csv('remaining_samples.csv')  # 選択されなかったサンプルを csv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意

print(selected_samples.corr()) # 相関行列の確認
