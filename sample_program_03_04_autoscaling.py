# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import pandas as pd

dataset = pd.read_csv('resin.csv', index_col=0, header=0)

deleting_variables = dataset.columns[dataset.std() == 0]  # 標準偏差が 0 の特徴量
dataset = dataset.drop(deleting_variables, axis=1)  # 標準偏差が 0 の特徴量の削除

autoscaled_dataset = (dataset - dataset.mean()) / dataset.std()  # 特徴量の標準化
autoscaled_dataset.to_csv('autoscaled_dataset.csv')

print('標準化後の平均値')
print(autoscaled_dataset.mean())
print('\n標準化後の標準偏差')
print(autoscaled_dataset.std())
