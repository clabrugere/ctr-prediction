from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    r2_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)

from helpers.trainer import train
from helpers.evaluation import calibration_error


def set_seed(seed=0):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def plot_training_curves(history):
    fig, axes = plt.subplots(nrows=2, figsize=(12, 6), sharex=True)
    
    sns.lineplot(x=np.arange(1, len(history["loss"]) + 1), y=history["loss"], label="loss", ax=axes[0])
    sns.lineplot(x=np.arange(1, len(history["val_loss"]) + 1), y=history["val_loss"], label="val loss", ax=axes[0])
    sns.lineplot(x=np.arange(1, len(history["mse"]) + 1), y=history["mse"], label="MSE", ax=axes[1])
    sns.lineplot(x=np.arange(1, len(history["val_mse"]) + 1), y=history["val_mse"], label="val MSE", ax=axes[1])
    
    return fig


# TODO current logic doesn't reinitialized weights each run
# use https://www.tensorflow.org/api_docs/python/tf/keras/models/clone_model instead
def benchmark(models, ds_train, ds_val, ds_test, epochs=10, lr=0.1, predict_batch_size=32, rounds=1):
    # unpack test data
    X_test = np.concatenate([x for x, _ in ds_test], axis=0)
    y_test = np.concatenate([y for _, y in ds_test], axis=0)

    results = {}

    for model in models:
        print(f"\nTraining {model.name}...")

        results[model.name] = defaultdict(list)
        results[model.name]["trainble_params"] = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        results[model.name]["total_params"] = np.sum([np.prod(v.shape) for v in model.variables])

        weights = model.get_weights()

        for round in range(rounds):
            print(f"Round {round+1}/{rounds}...")

            tf.keras.backend.clear_session()

            model.set_weights(weights)

            _, durations, _ = train(model, ds_train, ds_val, epochs=epochs, lr=lr, verbose=0, return_train_time=True)
            y_pred = model.predict(X_test, batch_size=predict_batch_size)

            results[model.name]["r2"].append(r2_score(y_test, y_pred))
            results[model.name]["mse"].append(brier_score_loss(y_test, y_pred))
            results[model.name]["logloss"].append(log_loss(y_test, y_pred))
            results[model.name]["ece"].append(calibration_error(y_test, y_pred))
            results[model.name]["auc_roc"].append(roc_auc_score(y_test, y_pred))
            results[model.name]["auc_pr"].append(average_precision_score(y_test, y_pred))
            results[model.name]["mean_epoch_duration"].append(np.mean(durations))

    return results


def results_to_df(results):
    result_df = {}
    for k, v in results.items():
        cols = list(v.keys())
        metrics = list(v.values())

        result_df[k] = metrics

    return pd.DataFrame.from_dict(result_df, orient="index", columns=cols)
