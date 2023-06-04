from torch import nn


class MLP(nn.Module):
    def __init__(self, dim_in, num_hidden, dim_hidden, dim_out=None, dropout=0.0):
        super().__init__()
        
        self.layers = nn.Sequential()
        for _ in range(num_hidden):
            self.layers.append(nn.Linear(dim_in, dim_hidden))
            self.layers.append(nn.BatchNorm1d(dim_hidden))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            dim_in = dim_hidden
        
        if dim_out:
            self.layers.append(nn.Linear(dim_hidden, dim_out))
            
    def forward(self, inputs):
        return self.layers(inputs)