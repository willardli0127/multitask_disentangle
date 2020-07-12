# -*- coding: utf-8 -*-
from time import time

from numpy import array, timedelta64
from pandas import DataFrame, date_range, merge, read_csv
from torch import cat, mean, Tensor
from torch.nn import (BCEWithLogitsLoss, GRU, Linear, Module, ModuleDict, 
                      MSELoss, ReLU, Sigmoid)
from torch.nn.init import orthogonal_, xavier_uniform_, zeros_
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


class SeqSet(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = [Y[i] for i in range(len(Y))]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], [self.Y[i][idx] for i in range(len(self.Y))]


class Net(Module):
    def __init__(self, n_features, n_embeddings, n_units):
        super(Net, self).__init__()
        self.n_features = n_features
        self.n_embeddings = n_embeddings
        self.n_units = n_units
        
        self.encoder = ModuleDict({
            'gru': GRU(self.n_features, self.n_units, 3, dropout=0.1, 
                       bidirectional=True, batch_first=True),
            'linear': Linear(2*self.n_units, self.n_embeddings)})
            
        self.decoder = ModuleDict({
            'gru': GRU(self.n_embeddings, self.n_units, 3, dropout=0.1,
                       bidirectional=True, batch_first=True),
            'linear': Linear(2*self.n_units, self.n_features)})
        
        self.decoder1 = ModuleDict({
            'gru': GRU(16, self.n_units, 3, dropout=0.1,
                       bidirectional=True, batch_first=True),
            'linear': Linear(2*self.n_units, 16)})
        
        self.decoder2 = ModuleDict({
            'gru': GRU(10, self.n_units, 3, dropout=0.1,
                       bidirectional=True, batch_first=True),
            'linear': Linear(2*self.n_units, 10)})
        
        self.decoder3 = ModuleDict({
            'gru': GRU(self.n_embeddings, self.n_units, 3, dropout=0.1,
                       bidirectional=True, batch_first=True),
            'linear': Linear(2*self.n_units, 1)})
        
        self.decoder4 = ModuleDict({
            'gru': GRU(self.n_embeddings, self.n_units, 3, dropout=0.1,
                       bidirectional=True, batch_first=True),
            'linear': Linear(2*self.n_units, 1)})
        
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        
    def forward(self, X):
        X, X_state = self.encoder['gru'](X)
        X = self.sigmoid(self.encoder['linear'](X[:, -1, :]))
        
        Y, Y_state = self.decoder['gru'](X.repeat(1, 30 ,1))
        Y = self.relu(self.decoder['linear'](Y))
        
        Y1, Y1_state = self.decoder1['gru'](X[:, :16].repeat(1, 30 ,1))
        Y1 = self.relu(self.decoder1['linear'](Y1[:, -1, :]))
        
        Y2, Y2_state = self.decoder2['gru'](X[:, 16:26].repeat(1, 30 ,1))
        Y2 = self.relu(self.decoder2['linear'](Y2[:, -1, :]))
        
        Y3, Y3_state = self.decoder3['gru'](X.repeat(1, 7 ,1))
        Y3 = self.relu(self.decoder3['linear'](Y3))
        
        Y4, Y4_state = self.decoder4['gru'](X.repeat(1, 7 ,1))
        Y4 = self.relu(self.decoder4['linear'](Y4))
        
        return [Y, Y1, Y2, Y3, Y4]


def net_init(net):
    for name, param in net.named_parameters():
        if 'bias' in name:
            zeros_(param)
        elif 'weight_ih' in name:
            orthogonal_(param)
        elif 'weight' in name:
            xavier_uniform_(param)

    return net


# def net_train(net, train_dl, epochs, loss_weights):
#     criterion = MSELoss()
#     optimizer = Adam(net.parameters())
#     for epoch in range(epochs):
#         start_time = time()
#         loss_epoch = []
#         for i, (X, Y) in enumerate(train_dl):
#             optimizer.zero_grad()
#             results = net(X)
#             loss = [criterion(results[j], Y[j]) for j in range(len(Y))]
#             loss_epoch.append(loss)
#             loss = sum([loss[j] * loss_weights[j] for j in range(len(Y))])
#             loss.backward()
#             optimizer.step()
#         loss_epoch = mean(Tensor(loss_epoch), 0)
#         print(epoch, loss_epoch, sum(loss_epoch * loss_weights),
#               (time() - start_time))

#     return net


def net_train(net, train_dl, epochs, loss_weights):
    criterion1 = BCEWithLogitsLoss()
    criterion2 = MSELoss()
    optimizer = Adam(net.parameters())
    weights_epoch = loss_weights
    for epoch in range(epochs):
        start_time = time()
        loss_epoch = []
        for i, (X, Y) in enumerate(train_dl):
            optimizer.zero_grad()
            results = net(X)
            
            loss0 = [8/34 * criterion1(results[0][:, :, :26], Y[0][:, :, :26]), 
                     26/34 * criterion2(results[0][:, :, 26:], Y[0][:, :, 26:])]
            loss0 = sum(loss0)
            loss1 = criterion1(results[1], Y[1])
            loss2 = criterion1(results[2], Y[2])
            loss3 = criterion2(results[3], Y[3])
            loss4 = criterion2(results[4], Y[4])
            loss = [loss0, loss1, loss2, loss3, loss4]
            
            loss_epoch.append(loss)
            loss = sum([loss[j] * weights_epoch[j] for j in range(len(Y))])
            
            loss.backward()
            optimizer.step()
        loss_epoch = mean(Tensor(loss_epoch), 0)
        print(epoch, loss_epoch, sum(loss_epoch * weights_epoch),
              (time() - start_time))
        weights_epoch = loss_epoch / sum(loss_epoch)
        print(weights_epoch)

    return net                


def sequence_split(values, n_steps_in, n_steps_out):
    X, Y = list(), list()
    for i in range(len(values)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        # check if we are beyond the dataset
        if out_end_ix > len(values):
            break
        # gather input and output parts of the pattern
        X.append(values[i:end_ix, :])
        Y.append(values[end_ix:out_end_ix, -2:])

    return Tensor(X), Tensor(Y)


path = '~/Downloads/'
n_steps_out = [7, 7]

grouped = read_csv(path + '1_2017-01-01_2020-02-28.csv',
                   usecols=['date', 'dept_id', 'model_id',
                            'e_pickup_count_1', 'e_return_count_1',
                            'create_count', 'tmp_order_count',
                            'day_0_order_ok_p', 'day_0_order_ok_r',
                            'pickup_count', 'return_count'],
                   parse_dates=['date'], infer_datetime_format=True)
dates = grouped['date'].sort_values().unique()
print(grouped)
earliest_date, latest_date = dates[0], dates[-1]
print(earliest_date, latest_date)

net = Net(34, (500 + 26), 500)
net = net_init(net)

loss_weights = Tensor([794/7256, 1334/7256, 1514/7256, 1807/7256, 1807/7256])

grouped = grouped.groupby(['dept_id','model_id'])
for idx, group in grouped:
    print(idx)
    group = group.set_index('date')[['dept_id', 'model_id',
                                      'e_pickup_count_1', 'e_return_count_1',
                                      'create_count', 'tmp_order_count',
                                      'day_0_order_ok_p', 'day_0_order_ok_r',
                                      'pickup_count', 'return_count']
                                    ].sort_index()
    
    dept = group['dept_id']
    model = group['model_id']
    group = group[['e_pickup_count_1', 'e_return_count_1', 'create_count', 
                    'tmp_order_count','day_0_order_ok_p', 'day_0_order_ok_r',
                    'pickup_count', 'return_count']]
    
    dept = DataFrame([list(f'{i:016b}') for i in dept.values], 
                      dept.index).astype('int')
    model = DataFrame([list(f'{i:010b}') for i in model.values], 
                      model.index).astype('int')
    dept_model = merge(dept, model, left_index=True, right_index=True)
    group = merge(dept_model, group, left_index=True, right_index=True)
    
    values = group.values
    X, Y = sequence_split(values, 30, max(n_steps_out))
    Y = [Y[:, :n_steps_out[i], i].unsqueeze(2) 
         for i in range(len(n_steps_out))]
    
    dl = DataLoader(SeqSet(X, [X, X[:, -1, :16], X[:, -1, 16:26], Y[0], Y[1]]))
    net = net_train(net, dl, 10, loss_weights)
    
