# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import pandas as pd
import numpy as np
from numpy import matlib

number_of_generating_samples = 10000  # 生成するサンプル数
desired_sum_of_components = 1 # 合計を指定する特徴量がある場合の、合計の値。例えば、この値を 100 にすれば、合計を 100 にできます

setting_of_generation = pd.read_csv('setting_of_generation.csv', index_col=0, header=0)

# 0 から 1 の間の一様乱数でサンプル生成
x_generated = np.random.rand(number_of_generating_samples, setting_of_generation.shape[1])

# 上限・下限の設定
x_upper = setting_of_generation.iloc[0, :]  # 上限値
x_lower = setting_of_generation.iloc[1, :]  # 下限値
x_generated = x_generated * (x_upper.values - x_lower.values) + x_lower.values  # 上限値から下限値までの間に変換

# 合計を 1 にする特徴量がある場合
if setting_of_generation.iloc[2, :].sum() != 0:
    for group_number in range(1, int(setting_of_generation.iloc[2, :].max()) + 1):
        variable_numbers = np.where(setting_of_generation.iloc[2, :] == group_number)[0]
        actual_sum_of_components = x_generated[:, variable_numbers].sum(axis=1)
        actual_sum_of_components_converted = np.matlib.repmat(np.reshape(actual_sum_of_components, (x_generated.shape[0], 1)) , 1, len(variable_numbers))
        x_generated[:, variable_numbers] = x_generated[:, variable_numbers] / actual_sum_of_components_converted
        deleting_sample_numbers, _ = np.where(x_generated > x_upper.values)
        x_generated = np.delete(x_generated, deleting_sample_numbers, axis=0)
        deleting_sample_numbers, _ = np.where(x_generated < x_lower.values)
        x_generated = np.delete(x_generated, deleting_sample_numbers, axis=0)

# 数値の丸め込みをする場合
if setting_of_generation.shape[0] >= 4:
    for variable_number in range(x_generated.shape[1]):
        x_generated[:, variable_number] = np.round(x_generated[:, variable_number], int(setting_of_generation.iloc[3, variable_number]))

# 保存
x_generated = pd.DataFrame(x_generated, columns=setting_of_generation.columns)
x_generated.to_csv('generated_samples.csv')  # 生成したサンプルをcsv ファイルに保存。同じ名前のファイルがあるときは上書きされるため注意
