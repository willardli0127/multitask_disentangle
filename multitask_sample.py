# -*- coding: utf-8 -*-
from math import sqrt
from time import time
from numpy import array, vstack
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch import cat, Tensor, save
from torch.nn import GRU, Linear, Module, MSELoss, ReLU
from torch.nn.init import orthogonal_, xavier_uniform_, zeros_
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

class SeqSet(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = [Y[i] for i in range(len(Y))]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], [self.Y[i][idx] for i in range(len(self.Y))]


class Net(Module):
    def __init__(self, n_steps_in, n_steps_out, n_features, n_cells):
        super(Net, self).__init__()
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.n_features = n_features
        self.n_cells = n_cells

        self.gru0 = GRU(n_features, self.n_cells, 4, dropout=0.1,
                        bidirectional=True, batch_first=True)
        self.gru1 = GRU(self.n_cells*2, self.n_cells, 3, dropout=0.1,
                        bidirectional=True, batch_first=True)
        self.gru2 = GRU(self.n_cells*2, self.n_cells, 3, dropout=0.1,
                        bidirectional=True, batch_first=True)

        self.linear1 = Linear(2*n_cells, 30)
        self.linear2 = Linear(2*n_cells, 30)

        self.linear1_2 = Linear(30, 1)
        self.linear2_2 = Linear(30, 1)

        self.relu = ReLU()

    def forward(self, X):
        X, X_state = self.gru0(X)
        X = self.relu(X)[:, -1, :]

        y1 = X.repeat(1, 7, 1)
        y1, y1_state = self.gru1(y1)
        y1 = self.relu(y1)
        y1 = self.relu(self.linear1(y1))
        y1 = self.relu(self.linear1_2(y1))

        y2 = X.repeat(1, 7, 1)
        y2, y2_state = self.gru2(y2)
        y2 = self.relu(y2)
        y2 = self.relu(self.linear2(y2))
        y2 = self.relu(self.linear2_2(y2))

        Y = [y1, y2]

        return Y


def net_init(net):
    for name, param in net.named_parameters():
        if 'bias' in name:
            zeros_(param)
        elif 'weight_hh' in name:
            xavier_uniform_(param)
        elif 'weight_ih' in name:
            orthogonal_(param)

    return net


def net_train(net, train_dl, epochs, loss_weights):
    criterion = MSELoss()
    optimizer = Adam(net.parameters())
    for epoch in range(epochs):
        start_time = time()
        loss_epoch = []
        for i, (X, Y) in enumerate(train_dl):
            optimizer.zero_grad()
            results = net(X)
            loss = [criterion(results[j], Y[j]) for j in range(len(Y))]
            loss_epoch.append(loss)
            loss = sum([loss[j] * loss_weights[j] for j in range(len(Y))])
            loss.backward()
            optimizer.step()
        loss_epoch = Tensor(loss_epoch)
        weight_epoch = loss_epoch.sum(0) / loss_epoch.sum()
        loss_epoch = loss_epoch.mean(0)
        print(epoch, loss_epoch, loss_weights, sum(loss_epoch * loss_weights),
              (time() - start_time))
        loss_weights = weight_epoch
    
    return net


def evaluate_forecasts(y, y_hat):
    print(len(y), len(y_hat))
    scores = list()
    # calculate an RMSE score for each day
    for i in range(y.shape[1]):
        # calculate mse
        mse = mean_squared_error(y[:, i, :], y_hat[:, i, :])

        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)

    # calculate overall RMSE
    s = 0
    for row in range(y_hat.shape[0]):
        for col in range(y_hat.shape[1]):
            s += (y[row, col, -3:] - y_hat[row, col, :])**2
    s = sum(s)
    score = sqrt(s / (y.shape[0] * y.shape[1]))

    return score, scores


def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.3f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


def online_evaluate(net, train_dl, test_dl, n_update, update_epochs,
                    loss_weight):
    X_hist, Y_hist = train_dl.dataset.X, train_dl.dataset.Y
    actuals, predictions = list(), list()
    n_targets = len(Y_hist)
    for j in range(n_targets):
        actuals.append(list())
        predictions.append(list())

    for i, (X, Y) in enumerate(test_dl):
        Y_hat = net.eval()(X)
        for j in range(n_targets):
            actuals[j].append(Y[j].numpy())
            predictions[j].append(Y_hat[j].detach().numpy())
            Y_hist[j] = cat((Y_hist[j], Y[j]), 0)

        X_hist = cat((X_hist, X), 0)
        update_dl = DataLoader(SeqSet(X_hist[-n_update:],
                                      [Y_hist[j][-n_update:]
                                       for j in range(n_targets)]))
        print('update', i)
        net = net_train(net, update_dl, update_epochs, loss_weight)

    score, scores = list(), list()
    for j in range(n_targets):
        actuals[j], predictions[j] = vstack(actuals[j]), vstack(predictions[j])
        print(actuals[j].shape, predictions[j].shape)
        score_, scores_ = evaluate_forecasts(actuals[j], predictions[j])
        score.append(score_)
        scores.append(scores_)

    return score, scores


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


def prepare_seq(values, n_train_len, n_steps_in, n_steps_out):
    X, Y = sequence_split(values, n_steps_in, max(n_steps_out))
    print(X.shape, Y.shape)

    Y = [Y[:, :n_steps_out[i], i] for i in range(len(n_steps_out))]
    Y = [Y[i].view(Y[i].shape[0], Y[i].shape[1], 1) for i in range(len(Y))]

    X_train, X_test = X[:n_train_len], X[n_train_len:]
    print(X_train.shape, X_test.shape)

    Y_train = [Y[i][:n_train_len]for i in range(len(Y))]
    Y_test = [Y[i][n_train_len:]for i in range(len(Y))]
    for i in range(len(Y)):
        print(Y_train[i].shape, Y_test[i].shape)

    return [X_train, Y_train], [X_test, Y_test]


def train_seq(values, n_steps_in, n_steps_out):
    X, Y = sequence_split(values, n_steps_in, max(n_steps_out))
    print(X.shape, Y.shape)

    Y = [Y[:, :n_steps_out[i], i] for i in range(len(n_steps_out))]
    Y = [Y[i].view(Y[i].shape[0], Y[i].shape[1], 1) for i in range(len(Y))]

    Y = [Y[i] for i in range(len(Y))]
    
    print(X.shape)
    for i in range(len(Y)):
        print(Y[i].shape)

    return [X, Y]


n_train_len = 90
n_steps_in = 30
n_steps_out = [7, 7]
n_cells = 500
n_train_epochs = 20
n_update = 7
n_update_epochs = 10
loss_weight = Tensor([(sum(n_steps_out) - n_steps_out[i])
                      for i in range(len(n_steps_out))]) / (
    (len(n_steps_out)-1) * (sum(n_steps_out)))

df = read_csv('~/Codes/Python/test/3.csv')
df['city_id'] = 1
df['dept_id'] = 513
df['model_id'] = 101

df = df[['holiday', 'amount_new', 'xallbeforeamount',
         'e_pickup_count_1', 'e_return_count_1', 'create_count',
         'tmp_order_count', 'day_0_order_ok_p', 'day_0_order_ok_r',
         'pickup_count', 'return_count']]
print(df)

values = df.values
print(values)
print(loss_weight)

n_features = values.shape[1]

seq_train, seq_test = prepare_seq(values, n_train_len, n_steps_in, n_steps_out)

train_dl = DataLoader(SeqSet(seq_train[0], seq_train[1]))
net = Net(n_steps_in, n_steps_out, n_features, n_cells)
net = net_init(net)

print(net)
print('train')
net = net_train(net, train_dl, n_train_epochs, loss_weight)

test_dl = DataLoader(SeqSet(seq_test[0], seq_test[1]))
score, scores = online_evaluate(
    net, train_dl, test_dl, n_update, n_update_epochs, loss_weight)
print(score, scores)
