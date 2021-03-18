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
import matplotlib.pyplot as plt

from collections import OrderedDict

from BestPractices.notebooks.CBFV.cbfv.composition import generate_features as gf

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

import time

start = time.time()

# 完全显示列表
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

PATH = os.getcwd()
train_path = os.path.join(PATH, 'BestPractices/data/cp_train.csv')
test_path = os.path.join(PATH, 'BestPractices/data/cp_test.csv')
val_path = os.path.join(PATH, 'BestPractices/data/cp_val.csv')

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_val = pd.read_csv(val_path)

rename_dict = {'Cp': 'target'}
df_train = df_train.rename(columns=rename_dict)
df_test = df_test.rename(columns=rename_dict)
df_val = df_val.rename(columns=rename_dict)

X_train_unscaled, y_train, formulae_train, skipped_train = gf(df_train,
                                                              elem_prop='oliynyk',
                                                              drop_duplicates=False,
                                                              extend_features=True,
                                                              sum_feat=True)
X_test_unscaled, y_test, formulae_test, skipped_test = gf(df_test,
                                                          elem_prop='oliynyk',
                                                          drop_duplicates=False,
                                                          extend_features=True,
                                                          sum_feat=True)
X_val_unscaled, y_val, formulae_val, skipped_val = gf(df_val,
                                                      elem_prop='oliynyk',
                                                      drop_duplicates=False,
                                                      extend_features=True,
                                                      sum_feat=True)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_unscaled)
X_test = scaler.fit_transform(X_test_unscaled)
X_val = scaler.fit_transform(X_val_unscaled)

X_train = normalize(X_train)
X_test = normalize(X_test)
X_val = normalize(X_val)

# Building a neural network
# define the network in PyTorch
class DenseNet(nn.Module):
    '''
    This implements a dynamically-bulit dense fully-connected neural network with leaky ReLU activation and optional
    dropout

    Parameters
    ---------------------
    input_dims = int
        Number of input features (required).
    hidden_dims: list of ints
        Number of hidden features, where each integer represents the number of hidden features in each subsequent
        hidden linear layer (optional, default=[64, 32]).
    output_dims = int
        Number of output features (optional, default=1).
    dropout: float
        the dropout value (optional, default=0.0).
    '''

    def __init__(self, input_dims, hidden_dims=[64, 32], output_dims=1, dropout=0.0):
        super.__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        self.dropout = dropout

        # Bulid a sub-block of linear network
        def fc_block(in_dim, out_dim, *args, **kwargs):
            return nn.Sequential(nn.Linear(in_dim, out_dim, *args, **kwargs),
                                 nn.Dropout(p=self.dropout),
                                 nn.LeakyReLU())

        # build overall networks architecture
        self.networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dims, self.hidden_dims[0]),
                    nn.Dropout(p=self.dropout),
                    nn.LeakyReLU())
                ]
        )

        hidden_layer_sizes = zip(self.hidden_dims[:-1], self.hidden_dims[1:])
        self.network.extend([
            fc_block(in_dim, out_dim) for in_dim, out_dim in hidden_layer_sizes]
        )

        self.network.extend([
            nn.Linear(hidden_dims[-1], output_dims)
        ])
    def forward(self, x):
        '''
        forward pass of the DenseNet model.

        :param x: torch.Tensor
                    A representation of the chemical compounds in the shape
                    (n_compounds, n_feats).
        :return:
        y: torch.Tensor
            The elements property prediction with the shape 1.
        '''

        for i, subnet in enumerate(self.network):
            x = subnet(x)

        y = x

        return y

CUDA_available = torch.cuda.is_available()
print(f'CUDA is available: {CUDA_available}')

if CUDA_available:
    compute_device = torch.device('cuda')
else:
    compute_device = torch.device('cpu')
print(f'Compute device for PyTorch: {compute_device}')

# defining the data loader and dataset structure
class CBFVDataLoader():
    '''
    :parameter:
    train_data: np.ndarray or pd.DataFrame or pd.Series
        name of csv file containing cif and properties
    test_data: np.ndarray or pd.DataFrame or pd.Series
        name of csv file containing cif and properties
    val_data: np.ndarray or pd.DataFrame or pd.Series
        name of csv file containing cif and properties
    batch_size: float, optional (default=42)
        Random seed for sampling the dataset. Only used if validation datais not given
    shuffle: bool, optional (default=True)
        Whether to shuffle the datasets or not
    '''
    def __init__(self, train_data, test_data, val_data,
                 batch_size=64, num_workers=1,random_state=42, shuffle=True, pin_memory=True):

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.shuffle = shuffle
        self.random_state = random_state

    def get_data_loaders(self, batch_size=1):
        train_dataset = CBFVDataLoader(self.train_data)
        test_dataset = CBFVDataLoader(self.test_data)
        val_dataset = CBFVDataLoader(self.val_data)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  pin_memory=self.pin_memory,
                                  shuffle=self.shuffle)

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 pin_memory=self.pin_memory,
                                 shuffle=self.shuffle)

        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                pin_memory=self.pin_memory,
                                shuffle=self.shuffle)

        return train_loader, test_loader, val_loader

class CBFVDataset(Dataset):
    '''
    get X and y from CBFV-based dataset.
    '''

    def __init__(self, dataset):
        self.data = dataset

        self.X = np.array(self.data[0])
        self.y = np.array(self.data[1])
        self.shape = [(self.X.shape), (self.y.shape)]

    def __str__(self):
        string = f'CBFVDataset with X.shape {self.X.shape}'
        return string

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[[idx], :]
        y = self.y[idx]

        X = torch.as_tensor(X)
        y = torch.as_tensor(np.array(y))

        return (X, y)

train_data = (X_train, y_train)
val_data = (X_val, y_val)
test_data = (X_test, y_test)

# Instantiate the DataLoader
batch_size = 128
data_loaders = CBFVDataLoader(train_data, val_data, test_data, batch_size=batch_size)
train_loader, val_loader, test_loader = data_loaders.get_data_loaders()



# Instantiate the DenseNet model
# Get input dimension size from the Dataset
example_data = train_loader.dataset.data[0]
input_dims = example_data.shape[-1]

model = DenseNet(input_dims, hidden_dims=[16], dropout=0.0)
print(model)



# defining the loss criterion and optimizer
# instantiate the loss criterion
criterion = nn.L1Loss()
print('Loss criterion: ')
print(criterion)

model = model.to(compute_device)
criterion = criterion.to(compute_device)
# instantiate the optimizer
optim_lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=optim_lr)
print('\nOptimizer: ')
print(optimizer)

# define some scaler function and helper function to evaluate and visualize model results
class Scaler():
    def __init__(self, data):
        self.data = torch.as_tensor(data)
        self.mean = torch.mean(self.data)
        self.std = torch.std(self.data)

    def scale(self, data):
        data = torch.as_tensor(data)
        data_scaled = (data - self.mean) / self.std
        return data_scaled

    def unscaled(self, data_scaled):
        data_scaled = torch.as_tensor(data_scaled)
        data = data_scaled * self.std + self.mean
        return data

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class MeanLogNormScaler():
    def __init__(self, data):
        self.data = torch.as_tenser(data)
        self.logdata = torch.log(self.data)
        self.mean = torch.mean(self.logdata)
        self.std = torch.std(self.logdata)

    def scale(self, data):
        data = torch.as_tensor(data)
        data_scaled = (torch.log(data) - self.mean) / self.std
        return data_scaled

    def unscale(self, data_scaled):
        data_scaled = torch.as_tensor(data_scaled) * self.std + self.mean
        data = torch.exp(data_scaled)
        return data

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def predict(model, data_loader):
    target_list = []
    pred_list = []

    model.eval()
    with torch.no_grad():
        for i, data_output in enumerate(data_loader):
            X, y_act = data_output
            X = X.to(compute_device,
                     dtype=data_type,
                     non_blocking=True)
            y_act = y_act.cpu().flatten.tolist()
            y_pred = model.forward(X).cpu().flatten().tolist()

        y_pred = target_scaler.unscale(y_pred).tolist()

        targets = y_act
        predictions = y_pred
        target_list.extend(targets)
        pred_list.extend(predictions)
    model.train()

    return target_list, pred_list

def evaluate(target, pred):
    r2 = r2_score(target, pred)
    mae = mean_absolute_error(target, pred)
    rmse = mean_squared_error(target, pred, squared=False)
    output = (r2, mae, rmse)

    return output

def print_scores(scores, label=''):
    r2, mae, rmse = scores

    print(f'{label} r2: {r2:0.4f}')
    print(f'{label} mae: {mae:0.4f}')
    print(f'{label} rmse: {rmse:0.4f}')

    return scores

def plot_pred_act(act, pred, model, reg_line=True, label=''):
    xy_max = np.max([np.max(act), np.max(pred)])
    plot = plt.figure(figsize=(6,6))
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

y_train = [data[1].numpy().tolist() for data in train_loader]
y_train = [item for sublist in y_train for item in sublist]
y_train = train_loader.dataset.y
target_scaler = MeanLogNormScaler(y_train)


# Training the neural networks
data_type = torch.float
epochs = 500
print_every = 20
plot_every = 50

for epoch in range(epochs):
    if epoch % print_every == 0 or epoch == epochs - 1:
        print(f'epoch: {epoch}')
    if epoch % plot_every == 0:
        target_train, pred_train = predict(model, train_loader)
    train_scores = evaluate(target_train, pred_train)
    print_scores(train_scores, label='train')
    target_val, pred_val = predict(model, val_loader)
    val_scores = evaluate(target_val, pred_val)
    print_scores(val_scores, label='val')
    plot_pred_act(target_val, pred_val, model, label='$\mathrm{C}_\mathrm{p}$ (J / mol K)')
    plt.show()

for i, data_output in enumerate(train_loader):
    X, y = data_output
    y = target_scaler.scale(y)
    X = X.to(compute_device,
             dtype=data_type,
             non_blocking=True)
    y = y.to(compute_device,
             dtype=data_type,
             non_blocking=True)

optimizer.zero_grad()
output = model.forward(X).flatten()
loss = criterion(output.view(-1), y.view(-1))
loss.backward()
optimizer.step()


target_val, pred_val = predict(model, val_loader)
scores = evaluate(target_val, pred_val)
print_scores(scores, label='val')
plot = plot_pred_act(target_val, pred_val, model, label='$\mathrm{C}_\mathrm{p}$ (J / mol K)')

# exporting pytorch model
save_dict = {'weights': model.state_dict(),
             'scaler_state': target_scaler.state_dict()}
print(save_dict)

pth_path = ('model_checkpoint.pth')  # .pth is commonly used as the file extension
torch.save(save_dict, pth_path)

print('time spent: %ds' % (time.time() - start))