# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import pandas as pd  # pandas の取り込み。一般的に pd と名前を省略して取り込みます

dataset = pd.read_csv('resin.csv', index_col=0, header=0)  # データセットの読み込み
#dataset = pd.read_csv('resin.csv', encoding='SHIFT-JIS', index_col=0, header=0)  # データセットの読み込み。日本語があるとき

print(dataset)  # 読み込んだデータセットを表示して確認
