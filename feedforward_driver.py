from torch import nn, optim
from models import MultiDimNN
from utils import train, score, plot_preds, generate_1D_data
from sklearn.model_selection import train_test_split
import numpy as np

# dataset settings
data_low = 0
data_high = 8 * np.pi
data_step = (data_high - data_low) / 10000
func = np.sin
# func = lambda x: [2 * elt + 5 for elt in x]

# training settings
lr = 0.01
n_epochs = 1000

# model settings
n_layers = 3
hidden_sizes = [128, 64, 128]

# create dataset
print('Generating dataset...')
X, y = generate_1D_data(func, low=data_low, high=data_high, step=data_step)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)
print('Dataset creation complete; creating model...')

# create model
model = MultiDimNN(1, n_layers, hidden_sizes)
print('Model creation complete')

# train
print('Training...')
train(model, X_train=X_train, y_train=y_train, n_epochs=n_epochs, lr=lr)
print('Training complete')

# evaluate & graph
print('Beginning evaluation...')
model.eval()
loss = score(model, X_test=X_test, y_test=y_test)
preds = model(X_test)
train_preds = model(X_train)
plot_preds(X_test, y_test, preds, X_train=X_train, y_train=y_train, y_train_pred=train_preds)