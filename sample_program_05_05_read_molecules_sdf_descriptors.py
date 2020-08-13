# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

property_name = 'logS'  # sdf ファイルの property の名前。property がない場合は何も書かないでください(property_name = '')

molecules = Chem.SDMolSupplier('molecules.sdf')  # sdf ファイルの読み込み
print('分子の数 :', len(molecules))

# 計算する記述子名の取得
descriptor_names = []
for descriptor_information in Descriptors.descList:
    descriptor_names.append(descriptor_information[0])
print('計算する記述子の数 :', len(descriptor_names))

# 記述子の計算
descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
# 分子ごとに、リスト型の変数 y に物性値を、descriptors に計算された記述子の値を、smiles に SMILES を追加
descriptors, y, smiles = [], [], []
for index, molecule in enumerate(molecules):
    print(index + 1, '/', len(molecules))
    if len(property_name):
        y.append(float(molecule.GetProp(property_name)))
    descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    smiles.append(Chem.MolToSmiles(molecule))
descriptors = pd.DataFrame(descriptors, index=smiles, columns=descriptor_names)
if len(property_name):
    y = pd.DataFrame(y, index=smiles, columns=[property_name])
    y = pd.DataFrame(y)  # Series のため列名の変更は別に
    y.columns = [property_name]
    descriptors = pd.concat([y, descriptors], axis=1)  # y と記述子を結合

# 保存
descriptors.to_csv('descriptors.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
