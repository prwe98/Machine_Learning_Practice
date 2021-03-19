'''
-*- coding: utf-8 -*-
@Name        : modeling_neural_network.py
@Time        : 2021/3/16 0016 10:13
@Author      : Xiaoyu Wu
@Institution : UESTC
'''

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport


# Load Data

PATH = os.getcwd()  # 路径返回当前进程目录
data_PATH = os.path.join(PATH, 'BestPractices/data/cp_data_demo.csv')  # 设置数据目录

df = pd.read_csv(data_PATH)
print(f'Original DataFrame shape: {df.shape}')

# Examine the data

print(df.head(10))
print("---------------------------------------------------------")
print(df.describe())
print("---------------------------------------------------------")

profile = ProfileReport(df.copy(), title='Pandas Profiling Report of Cp dataset', html={'style': {'full_width':True}})
profile.to_widgets()


# Rename the column names for brevity

df.columns
print(df.columns)
print("---------------------------------------------------------")
rename_dict = {'FORMULA': 'formula',
               'CONDITION: Temperature (K)': 'T',
               'PROPERTY: Heat Capacity (J/mol K)': 'Cp'}
df = df.rename(columns=rename_dict)
df.columns
print(df.columns)


# check for and remove 'NaN' values

df2 = df.copy()
bool_nans_formula = df2['formula'].isnull()
bool_nans_T = df2['T'].isnull()
bool_nans_Cp = df2['Cp'].isnull()

# drop the rows of DataFrame which contains NaNs
df2 = df2.drop(df2.loc[bool_nans_formula].index, axis=0)
df2 = df2.drop(df2.loc[bool_nans_T].index, axis=0)
df2 = df2.drop(df2.loc[bool_nans_Cp].index, axis=0)

print(f'DataFrame shape before dropping NaNs: {df.shape}')
print(f'DataFrame shape after dropping NaNs: {df2.shape}')

df3 = df.copy()
df3 = df3.dropna(axis=0, how='any')

print(f'DataFrame shape before dropping NaNs: {df.shape}')
print(f'DataFrame shape after dropping NaNs: {df3.shape}')

df = df3.copy()
print(df)