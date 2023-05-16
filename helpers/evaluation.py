import numpy as np
from sklearn.utils import assert_all_finite, check_consistent_length, column_or_1d


# taken from here: https://github.com/scikit-learn/scikit-learn/pull/11096/files
def calibration_error(
    y_true,
    y_prob,
    sample_weight=None,
    norm="l2",
    n_bins=10,
    strategy="uniform",
    pos_label=None,
    reduce_bias=True,
):
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    assert_all_finite(y_true)
    assert_all_finite(y_prob)
    check_consistent_length(y_true, y_prob, sample_weight)
    if any(y_prob < 0) or any(y_prob > 1):
        raise ValueError("y_prob has values outside of [0, 1] range")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            "Only binary classification is supported. " "Provided labels %s." % labels
        )

    if pos_label is None:
        pos_label = y_true.max()
    if pos_label not in labels:
        raise ValueError(
            "pos_label=%r is not a valid label: " "%r" % (pos_label, labels)
        )
    y_true = np.array(y_true == pos_label, int)

    norm_options = ("l1", "l2", "max")
    if norm not in norm_options:
        raise ValueError(f"norm has to be one of {norm_options}, got: {norm}.")

    remapping = np.argsort(y_prob)
    y_true = y_true[remapping]
    y_prob = y_prob[remapping]
    if sample_weight is not None:
        sample_weight = sample_weight[remapping]
    else:
        sample_weight = np.ones(y_true.shape[0])

    n_bins = int(n_bins)
    if strategy == "quantile":
        quantiles = np.percentile(y_prob, np.arange(0, 1, 1.0 / n_bins) * 100)
    elif strategy == "uniform":
        quantiles = np.arange(0, 1, 1.0 / n_bins)
    else:
        raise ValueError(
            f"Invalid entry to 'strategy' input. Strategy must be either "
            f"'quantile' or 'uniform'. Got {strategy} instead."
        )

    threshold_indices = np.searchsorted(y_prob, quantiles).tolist()
    threshold_indices.append(y_true.shape[0])
    avg_pred_true = np.zeros(n_bins)
    bin_centroid = np.zeros(n_bins)
    delta_count = np.zeros(n_bins)
    debias = np.zeros(n_bins)

    loss = 0.0
    count = float(sample_weight.sum())
    for i, i_start in enumerate(threshold_indices[:-1]):
        i_end = threshold_indices[i + 1]
        # ignore empty bins
        if i_end == i_start:
            continue
        delta_count[i] = float(sample_weight[i_start:i_end].sum())
        avg_pred_true[i] = (
            np.dot(y_true[i_start:i_end], sample_weight[i_start:i_end]) / delta_count[i]
        )
        bin_centroid[i] = (
            np.dot(y_prob[i_start:i_end], sample_weight[i_start:i_end]) / delta_count[i]
        )
        if norm == "l2" and reduce_bias:
            delta_debias = avg_pred_true[i] * (avg_pred_true[i] - 1) * delta_count[i]
            delta_debias /= count * delta_count[i] - 1
            debias[i] = delta_debias

    if norm == "max":
        loss = np.max(np.abs(avg_pred_true - bin_centroid))
    elif norm == "l1":
        delta_loss = np.abs(avg_pred_true - bin_centroid) * delta_count
        loss = np.sum(delta_loss) / count
    elif norm == "l2":
        delta_loss = (avg_pred_true - bin_centroid) ** 2 * delta_count
        loss = np.sum(delta_loss) / count
        if reduce_bias:
            loss += np.sum(debias)
        loss = np.sqrt(max(loss, 0.0))

    return loss
