# mysklearn/my_decision_tree.py

import math
from mysklearn import myutils

class MyDecisionTreeClassifier:
    """Decision Tree classifier using entropy and information gain.
       No pruning; standard greedy recursive splitting.
    """
    def __init__(self):
        self.tree = None

    # ---------- Utility functions ----------
    def entropy(self, values):
        """Compute entropy of a list of class labels."""
        freq = myutils.compute_frequencies(values)
        total = len(values)
        ent = 0
        for c in freq:
            p = freq[c] / total
            ent -= p * math.log2(p)
        return ent

    def info_gain(self, y, y_split_subsets):
        """Compute information gain from splitting y into subsets."""
        base_entropy = self.entropy(y)
        total = len(y)
        weighted_entropy = 0
        for subset in y_split_subsets:
            weighted_entropy += (len(subset) / total) * self.entropy(subset)
        return base_entropy - weighted_entropy

    def best_attribute(self, X, y):
        """Return attribute index with highest information gain."""
        n_features = len(X[0])
        best_gain = -1
        best_att = None
        
        for j in range(n_features):
            # Partition y by unique values in attribute j
            partitions = {}
            for i, row in enumerate(X):
                partitions.setdefault(row[j], []).append(y[i])

            gain = self.info_gain(y, list(partitions.values()))
            if gain > best_gain:
                best_gain = gain
                best_att = j
        
        return best_att

    # ---------- Recursive Tree Builder ----------
    def build_tree(self, X, y):
        """Recursive helper that returns a decision tree structure."""
        # Case 1: pure class
        if len(set(y)) == 1:
            return y[0]

        # Case 2: no attributes left
        if len(X[0]) == 0:
            # majority vote
            freq = myutils.compute_frequencies(y)
            return max(freq, key=freq.get)

        # Choose best attribute
        att = self.best_attribute(X, y)

        # Split rows by attribute value
        partitions = {}
        for i, row in enumerate(X):
            partitions.setdefault(row[att], {"X": [], "y": []})
            partitions[row[att]]["X"].append([val for k, val in enumerate(row) if k != att])
            partitions[row[att]]["y"].append(y[i])

        # Build subtree
        subtree = {"attribute": att, "branches": {}}
        for value, subset in partitions.items():
            subtree["branches"][value] = self.build_tree(subset["X"], subset["y"])

        return subtree

    # ---------- Public Methods ----------
    def fit(self, X_train, y_train):
        """Fit decision tree to training data."""
        self.tree = self.build_tree(X_train, y_train)

    def predict_one(self, row, tree):
        """Predict a single row using the tree."""
        # Leaf?
        if not isinstance(tree, dict):
            return tree

        att = tree["attribute"]
        value = row[att]

        # If unseen value, do majority class fallback
        if value not in tree["branches"]:
            # majority from leaf children
            leaves = []
            for branch in tree["branches"].values():
                if isinstance(branch, dict):
                    continue
                leaves.append(branch)
            if leaves:
                return max(leaves, key=leaves.count)
            else:
                return None

        # Traverse
        next_branch = tree["branches"][value]

        # Remove attribute from row for next level
        new_row = [v for i, v in enumerate(row) if i != att]
        return self.predict_one(new_row, next_branch)

    def predict(self, X_test):
        """Predict labels for X_test."""
        return [self.predict_one(row, self.tree) for row in X_test]
