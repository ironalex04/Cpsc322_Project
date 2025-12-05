import numpy as np
# utils.py



def compute_frequencies(values):
    """
    Computes frequency counts for a list of values.

    Parameters:
        values (list): Data values (numeric or string).

    Returns:
        dict: Mapping from value to frequency count.
    """
    freq = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    return freq


def safe_to_float(val):
    """
    Safely attempts to convert a value to float.

    Parameters:
        val (any): Input value.

    Returns:
        float or None: Numeric representation if valid, else None.
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def replace_missing_with_average(table, column_name):
    """
    Replaces 'NA' values in a given column with the column average.

    Parameters:
        table (MyPyTable): Table to modify.
        column_name (str): Column in which to replace missing values.
    """
    idx = table.column_names.index(column_name)
    vals = [safe_to_float(r[idx]) for r in table.data if r[idx] != "NA"]
    if not vals:
        return
    avg = sum(vals) / len(vals)
    for r in table.data:
        if r[idx] == "NA":
            r[idx] = avg

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds."""
    rng = np.random.default_rng(seed=random_state)

    label_indices = {}
    for idx, label in enumerate(y):
        label_indices.setdefault(label, []).append(idx)

    if shuffle:
        for label in label_indices:
            rng.shuffle(label_indices[label])

    folds = [[] for _ in range(n_splits)]
    for label, indices in label_indices.items():
        fold_sizes = [len(indices) // n_splits] * n_splits
        for i in range(len(indices) % n_splits):
            fold_sizes[i] += 1

        current = 0
        for fold_idx, size in enumerate(fold_sizes):
            start, stop = current, current + size
            folds[fold_idx].extend(indices[start:stop])
            current = stop

    if shuffle:
        rng.shuffle(folds)

    all_indices = np.arange(len(X))
    stratified_folds = []
    for fold in folds:
        test_indices = np.array(fold)
        train_indices = np.setdiff1d(all_indices, test_indices)
        stratified_folds.append((train_indices.tolist(), test_indices.tolist()))

    return stratified_folds

from tabulate import tabulate

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate a classification."""
    n_labels = len(labels)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((n_labels, n_labels), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        i = label_to_index[yt]
        j = label_to_index[yp]
        matrix[i][j] += 1

    return matrix.tolist()


def accuracy_score(y_true, y_pred, normalize=True):
    """Compute classification accuracy."""
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    total = len(y_true)
    if normalize:
        return correct / total if total > 0 else 0.0
    else:
        return correct


def classification_report(y_true, y_pred, labels=None, output_dict=False):
    """Simple classification report with precision, recall, F1, and support."""
    if labels is None:
        labels = list(dict.fromkeys(list(y_true) + list(y_pred)))
    
    metrics = {}
    support_counts = {label: 0 for label in labels}
    for yt in y_true:
        support_counts[yt] += 1

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support_counts[label]
        }

    macro_precision = sum(metrics[l]["precision"] for l in labels) / len(labels)
    macro_recall = sum(metrics[l]["recall"] for l in labels) / len(labels)
    macro_f1 = sum(metrics[l]["f1-score"] for l in labels) / len(labels)
    total_support = sum(metrics[l]["support"] for l in labels)

    metrics["macro avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1-score": macro_f1,
        "support": total_support
    }

    if output_dict:
        return metrics

    rows = []
    for label in labels + ["macro avg"]:
        m = metrics[label]
        rows.append([label, m["precision"], m["recall"], m["f1-score"], m["support"]])

    return tabulate(rows, headers=["class", "precision", "recall", "f1-score", "support"],
                    floatfmt=".3f", tablefmt="grid")

