'''
-*- coding: utf-8 -*-
@Name        : modeling_neural_network.py
@Time        : 2021/3/16 0016 10:13
@Author      : Xiaoyu Wu
@Institution : UESTC
'''

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# set random seed to ensure reproducibility across runs
RNG_SEED = 42
np.random.seed(seed=RNG_SEED)

# Load the pre-processed dataset
PATH = os.getcwd()
data_PATH = os.path.join(PATH, 'BestPractices/data/cp_data_cleaned.csv')

df = pd.read_csv(data_PATH)
print(f'Full DataFrame shape: {df.shape}')


X = df[['formula', 'T']]
Y = df['Cp']

print(f'shape of X: {X.shape}')
print(f'shape of Y: {Y.shape}')


# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RNG_SEED)

# Split data's scale X_train and X_test
print(X_train.shape)
print(X_test.shape)

# unique() return the value without the repeat and in order

num_rows = len(X_train)
print(f'There are in total {num_rows} rows in X_train DataFrame')

num_unique_formula = len(X_train['formula'].unique())
print(f'But there are only {num_unique_formula} unique formula!\n')

print('Unique formulae and their number of occurances in X_train DataFrame:')
print(X_train['formula'].value_counts(), '\n')   # .value_counts()可以计算'formula'有多少重复值
print(X_test['formula'].value_counts())


# Splitting data, cautiously (manually)

unique_formulae = X['formula'].unique()
print(f'{len(unique_formulae)} unique formulae:\n{unique_formulae}')

# set a random seed
np.random.seed(seed=RNG_SEED)

# store a list of all unique formulae
all_formulae = unique_formulae.copy()


# define the proportional size of the dataset split
val_size = 0.20
test_size = 0.10
train_size = 1 - val_size -test_size

# Calculate the number of samples in each dataset split
num_val_samples = int(round(val_size * len(unique_formulae)))  # round(数，位数)四舍五入保留小数点后几位函数
num_test_samples = int(round(test_size * len(unique_formulae)))  # round(数，位数)四舍五入保留小数点后几位函数
num_train_samples = int(round(train_size * len(unique_formulae)))  # round(数，位数)四舍五入保留小数点后几位函数

# randomly choose the formulate for the validation dataset, and remove those from the unique formulae list
val_formulae = np.random.choice(all_formulae, size=num_val_samples, replace=False)
all_formulae = [f for f in all_formulae if f not in val_formulae]
# numpy.random.choice(a, size=None, replace=True, p=None)
# 从a(只要是np.array都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
# replace:True表示可以取相同数字，False表示不可以取相同数字
# 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。

test_formulae = np.random.choice(all_formulae, size=num_test_samples, replace=False)
all_formulae = [f for f in all_formulae if f not in test_formulae]

# The remaining formulae will used for the training dataset
train_formulae = all_formulae.copy()

print('Number of training formulae:', len(train_formulae))
print('Number of test formulae:', len(test_formulae))
print('Number of value formulae:', len(val_formulae))


# splitting using the formulae lists above
df_train = df[df['formula'].isin(train_formulae)]
df_test = df[df['formula'].isin(test_formulae)]
df_val = df[df['formula'].isin(val_formulae)]

train_formulae = set(df_train['formula'].unique())
test_formulae = set(df_test['formula'].unique())
val_formulae = set(df_val['formula'].unique())
# set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等

common_formulae1 = train_formulae.intersection(test_formulae)
common_formulae2 = train_formulae.intersection(val_formulae)
common_formulae3 = test_formulae.intersection(val_formulae)
# x.intersection(y) 方法用于返回两个或更多(x and y)集合中都包含的元素，即交集

print(f'# of common formulae in intersection 1: {len(common_formulae1)}')
print(f'# of common formulae in intersection 2: {len(common_formulae2)}')
print(f'# of common formulae in intersection 3: {len(common_formulae3)}')


# save split datasets into csv
PATH = os.getcwd()
train_path = os.path.join(PATH, './cp_train.csv')
test_path = os.path.join(PATH, './cp_test.csv')
val_path = os.path.join(PATH, './cp_val.csv')

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)
df_val.to_csv(val_path, index=False)
