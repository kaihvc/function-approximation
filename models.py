'''
    This module holds the neural network models I'll be using to perform
        these function approximations.
'''
from torch import nn

class MultiDimNN(nn.Module):
    # def __init__(self, input_dim, hidden_size):
    #     super().__init__()
    #     self.linear_1 = nn.Linear(input_dim, hidden_size)
    #     self.linear_2 = nn.Linear(hidden_size, hidden_size)
    #     self.output_layer = nn.Linear(hidden_size, 1)
    #     self.activation = nn.Tanh()
    #     self.double()

    def __init__(self, input_dim, n_layers, hidden_sizes, activation = None):
        super().__init__()
        assert len(hidden_sizes) == n_layers, "n_layers must match length of given dimensions"
        hidden_sizes = [input_dim] + hidden_sizes

        self.linears = []
        for l in range(n_layers):
            self.linears.append(nn.Linear(hidden_sizes[l], hidden_sizes[l + 1]).double())
        
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        self.activation = nn.Tanh() if activation is None else activation
        self.double()


    # def forward(self, X):
    #     output = self.activation(self.linear_1(X))
    #     output = self.activation(self.linear_2(output))
    #     output = self.output_layer(output)
    #     return output

    def forward(self, X):
        output = X
        for layer in self.linears:
            output = self.activation(layer(output))

        output = self.output_layer(output)

        return output
