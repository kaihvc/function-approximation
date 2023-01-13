from torch import nn, optim
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train(model: nn.Module, X_train, y_train, n_epochs: int = 20, lr: float = 0.01):
    # create loss fn. & choose an optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # set model to training mode - allows modification of weights
    model.train()

    # actual training loop
    for epoch in range(n_epochs):
        # zero all gradients so we're not using previous epochs
        optimizer.zero_grad()

        # get predictions & loss
        y_pred = model(X_train)
        loss = criterion(y_pred.squeeze(), y_train)

        # report
        print(f'Epoch {epoch + 1}: train loss {loss.item()}')

        # backpropagate loss (calculates gradients - *doesn't update*)
        loss.backward()

        # make updates to weights
        optimizer.step()


def score(model: nn.Module, X_test, y_test):
    criterion = nn.MSELoss()

    model.eval()
    y_pred = model(X_test)

    return criterion(y_pred.squeeze(), y_test)


def plot_preds(X_test, y_test, y_pred, X_train = None, y_train = None, y_train_pred = None):
    x_plot = X_test.detach().numpy()
    y_test_plot = y_test.detach().numpy()
    y_pred_plot = y_pred.detach().numpy()
    _, ax = plt.subplots(figsize=(16, 8))
    ax.plot(x_plot, y_test_plot, color='tab:blue', label='test data')
    ax.plot(x_plot, y_pred_plot, color='tab:red', label='prediction')
    if X_train is not None and y_train is not None and y_train_pred is not None:
        ax.plot(X_train.detach().numpy(), y_train.detach().numpy(), color='tab:orange', label='train data')
        ax.plot(X_train.detach().numpy(), y_train_pred.detach().numpy(), color='tab:green', label='train data prediction')
    ax.legend()

    plt.show()


class FuncApproxDataSet(Dataset):
    def __init__(self, func, low: float = 0, high: float = 100, step: float = 0.1):
        self.x = torch.tensor(np.arange(low, high, step=step))
        self.y = torch.tensor(func(self.x))

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def generate_1D_data(func, low: float = 0.0, high: float = 100, step: float = 0.1):
    x = torch.tensor(np.arange(start=low, stop=high, step=step)).unsqueeze(1)
    y = torch.tensor(func(x)).squeeze()

    return x, y