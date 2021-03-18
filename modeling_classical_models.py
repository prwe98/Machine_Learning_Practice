import time
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from CBFV.cbfv.composition import generate_features as gf

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR, LinearSVR

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.preprocessing import normalize

from Data_scaling import X_train, X_val, X_test, scaler
from data_featurization import Y_train, Y_val, Y_test, df_train, df_test, df_val

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def instantiate_model(model_name):
    model = model_name()
    return model


def fit_model(model, X_train, Y_train):
    start_time = time()
    model = instantiate_model(model)
    model.fit(X_train, Y_train.astype('int'))
    fit_time = time() - start_time
    return model, fit_time


def evaluate_model(model, X, Y_act):
    Y_pred = model.predict(X)
    r2 = r2_score(Y_act, Y_pred)
    mae = mean_absolute_error(Y_act, Y_pred)
    rmse_val = mean_squared_error(Y_act, Y_pred, squared=False)
    return r2, mae, rmse_val


def fit_evaluate_model(model, model_name, X_train, Y_train, X_val, Y_act_val):
    model, fit_time = fit_model(model, X_train, Y_train.astype('int'))
    r2_train, mae_train, rmse_train = evaluate_model(model, X_train, Y_train)
    r2_val, mae_val, rmse_val = evaluate_model(model, X_val, Y_act_val)
    result_dict = {'model_name': model_name,
                   'model_name_pretty': type(model).__name__,
                   'model_params': model.get_params(),
                   'fit time': fit_time,
                   'r2_train': r2_train,
                   'mae_train': mae_train,
                   'rmse_train': rmse_train,
                   'r2_val': r2_val,
                   'mae_val': mae_val,
                   'rmse_val': rmse_val}
    return model, result_dict


def append_result_df(df, result_dict):
    df_result_appended = df.append(result_dict, ignore_index=True)
    return df_result_appended


def append_model_dict(dic, model_name, model):
    dic[model_name] = model
    return dic


# build an empty DataFrame to store model result

df_classics = pd.DataFrame(columns=['model_name',
                                    'model_name_pretty',
                                    'model_params',
                                    'fit_time',
                                    'r2_train',
                                    'mae_train',
                                    'rmse_train',
                                    'r2_val',
                                    'mae_val',
                                    'rmse_val'])
df_classics
# Define the models
# build a dictionary of model names
classic_model_names = OrderedDict({'dumr': DummyRegressor,
                                   'rr': Ridge,
                                   'abr': AdaBoostRegressor,
                                   'gbr': GradientBoostingRegressor,
                                   'rfr': RandomForestRegressor,
                                   'etr': ExtraTreesRegressor,
                                   'svr': SVR,
                                   'lsvr': LinearSVR,
                                   'knr': KNeighborsRegressor})

# instantiate and fit the models
# instantiate a dictionary to store the model objects
classic_models = OrderedDict()
start_time = time()

# loop through each model type, fit and predict, and evaluate and store results
for model_name, model in classic_model_names.items():
    print(f'Now fitting and evaluating model {model_name}: {model.__name__}')
    model, result_dict = fit_evaluate_model(model, model_name, X_train, Y_train.astype('int'), X_val, Y_val)
    df_classics = append_result_df(df_classics, result_dict)
    classic_models = append_model_dict(classic_models, model_name, model)

end_time = time() - start_time
print(f'Finished fitting {len(classic_models)} models, total time: {end_time:0.2f} s')

# result
# sort in order of increasing validation r2 score
df_classics = df_classics.sort_values('r2_val', ignore_index=True)
print(df_classics)


# evaluating model performance on validation dataset
def plot_pred_act(act, pred, model, reg_line=True, label=''):
    xy_max = np.max([np.max(act), np.max(pred)])

    plot = plt.figure(figsize=(6, 6))
    plt.plot(act, pred, 'o', ms=9, mec='k', mfc='silver', alpha=0.4)
    plt.plot([0, xy_max], [0, xy_max], 'k--', label='ideal')
    if reg_line:
        polyfit = np.polyfit(act, pred, deg=1)
        reg_ys = np.poly1d(polyfit)(np.unique(act))
        plt.plot(np.unique(act), reg_ys, alpha=0.8, label='linear fit')
    plt.axis('scaled')
    plt.xlabel(f'Actual {label}')
    plt.ylabel(f'Predicted {label}')
    plt.title(f'{type(model).__name__}, r2: {r2_score(act, pred):0.4f}')
    plt.legend(loc='upper left')

    return plot


for row in range(df_classics.shape[0]):
    model_name = df_classics.iloc[row]['model_name']

    model = classic_models[model_name]
    Y_act_val = Y_val
    Y_pred_val = model.predict(X_val)

    plot = plot_pred_act(Y_act_val, Y_pred_val, model, reg_line=True, label='$\mathrm{c}_\mathrm{p}$ (J / mol k)')

# re_training the best-performance model on combined train + validation dadasets
# find the best
best_row = df_classics.iloc[-1, :].copy()

# get the model type and model parameters
model_name = best_row['model_name']
model_params = best_row['model_params']

# instantiate the model again using the parameters
model = classic_model_names[model_name](**model_params)
print(model)

# concatenate the train and validation dataset together
X_train_new = np.concatenate((X_train, X_val), axis=0)
Y_train_new = np.concatenate((Y_train, Y_val), axis=0)  # np.concatenate()用于一维数组的拼接

start_time = time()

model.fit(X_train_new, Y_train_new)

end_time = time() - start_time
print(f'Finish fitting best model, total time {end_time:0.2f} s')

# test re-trained model
Y_act_test = Y_test
Y_pred_test = model.predict(X_test)

r2, mae, rmse, = evaluate_model(model, X_test, Y_test)
print(f'r2: {r2:0.4f}')
print(f'mae: {mae:0.4f}')
print(f'rmse: {rmse:0.4f}')

plot_2 = plot_pred_act(Y_act_test, Y_pred_test, model, reg_line=True, label='$\mathrm{c}_\mathrm{p}$ (J / mol k)')

# effect of train/validation/test/dataset split
X_train_unscaled, Y_train, formulae_train, skipped_train = gf(df_train,
                                                              elem_prop='oliynyk',
                                                              drop_duplicates=False,
                                                              extend_features=True,
                                                              sum_feat=True)
X_val_unscaled, Y_val, formulae_val, skipped_val = gf(df_val,
                                                      elem_prop='oliynyk',
                                                      drop_duplicates=False,
                                                      extend_features=True,
                                                      sum_feat=True)
X_test_unscaled, Y_test, formulae_test, skipped_test = gf(df_test,
                                                          elem_prop='oliynyk',
                                                          drop_duplicates=False,
                                                          extend_features=True,
                                                          sum_feat=True)

X_train_original = X_train_unscaled.copy()
X_val = X_val_unscaled.copy()
X_test = X_test_unscaled.copy()

Y_train_original = Y_train.copy()

splits = range(10)
df_splits = pd.DataFrame(columns=['split',
                                  'r2_train',
                                  'mae_train',
                                  'rmse_train',
                                  'r2_val',
                                  'mae_val',
                                  'rmse_val'])

for split in splits:
    print(f'Fitting and evaluating random split {split}')
    X_train = X_train_original.sample(frac=0.7, random_state=split)
    Y_train = Y_train_original[X_train.index]

    X_train = normalize(scaler.fit_transform(X_train))
    X_val = normalize(scaler.fit_transform(X_val))
    X_test = normalize(scaler.fit_transform(X_test))

    model = AdaBoostRegressor()
    model.fit(X_train, Y_train)
    Y_act_val = Y_val
    Y_pred_val = model.predict(X_val)

    r2_train, mae_train, rmse_train = evaluate_model(model, X_train, Y_train)
    r2_val, mae_val, rmse_val = evaluate_model(model, X_val, Y_val)
    result_dict = {'split': split,
                   'r2_train': r2_train,
                   'mae_train': mae_train,
                   'rmse_train': rmse_train,
                   'r2_val': r2_val,
                   'mae_val': mae_val,
                   'rmse_val': rmse_val}

    df_splits = append_result_df(df_splits, result_dict)

df_splits['split'] = df_splits['split'].astype(int)
print(df_splits)

plot_3 = df_splits.plot('split', ['r2_train', 'r2_val'], kind='bar')
plt.title(f'Performance of {type(model).__name__}\nwith {len(splits)} different data splits')
plt.ylim((0.5, 1.0))
plt.xlabel('Split #')
plt.ylabel('$r^2$')
plt.legend(loc='lower right', framealpha=0.9)
plot_3.show()

plot_4 = df_splits.plot('split', ['mae_train', 'mae_val'], kind='bar')
plt.title(f'Performance of {type(model).__name__}\nwith {len(splits)} different data splits')
plt.xlabel('Split #')
plt.ylabel('MAE in $\mathrm{c}_\mathrm{p}$ (J / mol K)')
plt.legend(loc='lower right', framealpha=0.9)
plot_4.show()

avg_r2_val = df_splits['r2_val'].mean()  # mean()在pandas中求取平均值
avg_mae_val = df_splits['mae_val'].mean()

print(f'Average validation r2: {avg_r2_val:0.4f}')
print(f'Average validation mae: {avg_mae_val:0.4f}')
