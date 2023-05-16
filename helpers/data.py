import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np


def process_dataset(df):
    labels = df["label"]
    input = pd.DataFrame(dict(zip(df["inputs"].index, df["inputs"].values))).T

    return input, labels


def df_to_dataset(X, y, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def get_splits(X, y, test_size=0.2, val_size=0.1, batch_size=32):
    # prepare splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, shuffle=False)

    assert len(set(X_train.index) & set(X_val.index) & set(X_test.index)) == 0

    # compute number of unique objects to embed
    num_embedding = np.max(X_train)

    num_features = X_train.shape[1]

    # prepare tf.Datasets
    ds_train = df_to_dataset(X_train, y_train, batch_size=batch_size)
    ds_val = df_to_dataset(X_val, y_val, batch_size=batch_size)
    ds_test = df_to_dataset(X_test, y_test, shuffle=False, batch_size=batch_size)

    return ds_train, ds_val, ds_test, num_features, num_embedding
