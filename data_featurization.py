'''
-*- coding: utf-8 -*-
@Name        : modeling_neural_network.py
@Time        : 2021/3/16 0016 10:13
@Author      : Xiaoyu Wu
@Institution : UESTC
'''

# using CBFVs featurizing materials composition data
import os
import numpy as np
import pandas as pd
from CBFV.cbfv.composition import generate_features as gf

RNG_SEED = 42
np.random.seed(RNG_SEED)

# loading data
PATH = os.getcwd()
train_path = os.path.join(PATH, './cp_train.csv')
test_path = os.path.join(PATH, './cp_test.csv')
val_path = os.path.join(PATH, './cp_val.csv')

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_val = pd.read_csv(val_path)

# sub-sampling your data
df_train_sampled = df_train.sample(n=2000, random_state=RNG_SEED)
df_test_sampled = df_test.sample(n=200, random_state=RNG_SEED)
df_val_sampled = df_val.sample(n=200, random_state=RNG_SEED)
# 通过种子随机random_state来sub-sampling 数据，由3000+缩小到2000


print('DataFrame column names before renaming:')
print(df_train.columns)
print(df_test.columns)
print(df_val.columns)

rename_dict = {'Cp': 'target'}
df_train = df_train.rename(columns=rename_dict)
df_test = df_test.rename(columns=rename_dict)
df_val = df_val.rename(columns=rename_dict)

df_train_sampled = df_train_sampled.rename(columns=rename_dict)
df_test_sampled = df_test_sampled.rename(columns=rename_dict)
df_val_sampled = df_val_sampled.rename(columns=rename_dict)

print('\nDataFrame column names after renaming:')
print(df_train.columns)
print(df_test.columns)
print(df_val.columns)

X_train_unscaled, Y_train, formulae_train, skipped_train = gf(df_train_sampled,
                                                              elem_prop='oliynyk',
                                                              drop_duplicates=False,
                                                              extend_features=True,
                                                              sum_feat=True)
X_val_unscaled, Y_val, formulae_val, skipped_val = gf(df_val_sampled,
                                                      elem_prop='oliynyk',
                                                      drop_duplicates=False,
                                                      extend_features=True,
                                                      sum_feat=True)
X_test_unscaled, Y_test, formulae_test, skipped_test = gf(df_test_sampled,
                                                          elem_prop='oliynyk',
                                                          drop_duplicates=False,
                                                          extend_features=True,
                                                          sum_feat=True)

X_train_unscaled.head()
print(X_train_unscaled.head)
print(X_train_unscaled.shape)