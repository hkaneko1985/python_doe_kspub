# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import pandas as pd
import matplotlib.pyplot as plt  # matplotlib の pyplot の読み込み。一般的に plt と名前を省略して取り込みます

number_of_variable = 0  # ヒストグラムを描画する特徴量の番号。Python では 0 から順番が始まるため注意しましょう
number_of_bins = 10  # ビンの数

dataset = pd.read_csv('resin.csv', index_col=0, header=0)  # データセットの読み込み

plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.hist(dataset.iloc[:, number_of_variable], bins=number_of_bins)  # ヒストグラムの作成
plt.xlabel(dataset.columns[number_of_variable])  # 横軸の名前
plt.ylabel('frequency')  # 縦軸の名前
plt.show()  # 以上の設定において、グラフを描画
