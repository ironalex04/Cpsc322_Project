import random
import numpy as np
from collections import Counter
from mysklearn.my_decision_tree import MyDecisionTreeClassifier

class MyRandomForestClassifier:
    """Custom Random Forest using our CPSC 322 Decision Tree."""

    def __init__(self, n_estimators=25, max_features="sqrt", bootstrap_ratio=0.7, random_state=1):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap_ratio = bootstrap_ratio
        self.random_state = random_state
        self.trees = []
        self.feature_subsets = []
        random.seed(random_state)

    def _select_features(self, n_total):
        """Randomly select subset of features for split."""
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_total)))
        else:
            k = max(1, int(self.max_features))
        return random.sample(range(n_total), k)

    def _bootstrap(self, X, y):
        """Create bootstrap sample."""
        n = len(X)
        size = int(self.bootstrap_ratio * n)
        indices = [random.randrange(n) for _ in range(size)]
        return [X[i] for i in indices], [y[i] for i in indices]

    def fit(self, X_train, y_train):
        """Train ensemble of decision trees."""
        self.trees = []
        self.feature_subsets = []
        n_total = len(X_train[0])

        for _ in range(self.n_estimators):
            features = self._select_features(n_total)
            X_samp, y_samp = self._bootstrap(X_train, y_train)

            X_samp_sub = [[row[i] for i in features] for row in X_samp]

            dt = MyDecisionTreeClassifier()
            dt.fit(X_samp_sub, y_samp)

            self.trees.append(dt)
            self.feature_subsets.append(features)

    def predict(self, X_test):
        """Majority vote prediction."""
        predictions = []
        for x in X_test:
            votes = []
            for dt, features in zip(self.trees, self.feature_subsets):
                x_sub = [x[i] for i in features]
                votes.append(dt.predict([x_sub])[0])
            predictions.append(Counter(votes).most_common(1)[0][0])
        return predictions
