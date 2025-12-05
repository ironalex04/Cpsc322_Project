# mysklearn/my_naive_bayes.py

import math
from mysklearn import myutils

class MyNaiveBayesClassifier:
    """
    Naive Bayes classifier with Laplace smoothing.
    Works for categorical data (what we use in LAPD crime classification).
    """

    def __init__(self, laplace=1):
        self.laplace = laplace  # smoothing constant
        self.priors = None      # {label: P(label)}
        self.conditionals = None  # {label: {feature_index: {value: P(value|label)}}}
        self.label_values = None     # list of possible labels
        self.feature_values = None   # list of sets of possible values per feature

    def fit(self, X_train, y_train):
        """Compute priors and conditional probabilities with Laplace smoothing."""
        # ---- Priors ----
        freq_y = myutils.compute_frequencies(y_train)
        n = len(y_train)
        self.priors = {c: freq_y[c] / n for c in freq_y}
        self.label_values = list(self.priors.keys())

        # Track possible feature values for smoothing and unseen values
        num_features = len(X_train[0])
        self.feature_values = [set() for _ in range(num_features)]
        for row in X_train:
            for j, val in enumerate(row):
                self.feature_values[j].add(val)

        #  Conditionals
        self.conditionals = {c: {j: {} for j in range(num_features)} for c in self.priors}

        # Calculate conditional probabilities P(x|c)
        for c in self.priors:
            # rows for class c
            class_rows = [X_train[i] for i in range(n) if y_train[i] == c]

            for j in range(num_features):
                col_vals = [row[j] for row in class_rows]
                freq = myutils.compute_frequencies(col_vals)
                total = len(col_vals)

                # number of unique categories for smoothing
                k = len(self.feature_values[j])

                for val in self.feature_values[j]:
                    # Apply Laplace smoothing: P = (count + laplace) / (total + laplace * k)
                    count = freq.get(val, 0)
                    prob = (count + self.laplace) / (total + self.laplace * k)
                    self.conditionals[c][j][val] = prob

    def predict(self, X_test):
        """Predict labels for test instances using naive Bayes rule."""
        y_pred = []

        for row in X_test:
            class_probs = {}

            for c in self.priors:
                # Start with log prior to avoid underflow on multiplication
                log_prob = math.log(self.priors[c])

                for j, val in enumerate(row):
                    # If unseen value, treat as uniform smoothing
                    prob = self.conditionals[c][j].get(
                        val,
                        1 / (len(self.feature_values[j]) + self.laplace)
                    )
                    log_prob += math.log(prob)

                class_probs[c] = log_prob

            best = max(class_probs, key=class_probs.get)
            y_pred.append(best)

        return y_pred
