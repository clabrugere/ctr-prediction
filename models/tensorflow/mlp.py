from keras import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, ReLU


class MLP(Sequential):
    def __init__(self, num_hidden, dim_hidden, dim_out=None, batch_norm=True, dropout=0.0, name="MLP"):
        layers = []
        for _ in range(num_hidden - 1):
            layers.append(Dense(dim_hidden))

            if batch_norm:
                layers.append(BatchNormalization())

            layers.append(ReLU())
            layers.append(Dropout(dropout))

        if dim_out:
            layers.append(Dense(dim_out))
        else:
            layers.append(Dense(dim_hidden))

        super().__init__(layers, name=name)
