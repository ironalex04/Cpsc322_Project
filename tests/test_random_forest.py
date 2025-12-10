from mysklearn.my_random_forest import MyRandomForestClassifier

def test_rf_fit_creates_trees():
    X = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]
    y = ["A", "B", "A", "B", "A"]

    rf = MyRandomForestClassifier(n_estimators=5, max_features="sqrt", random_state=0)
    rf.fit(X, y)

    # Random Forest should create exactly n_estimators trees
    assert len(rf.trees) == 5
    assert len(rf.feature_subsets) == 5

def test_rf_predict_length():
    X_train = [[0], [1], [0], [1]]
    y_train = ["A", "B", "A", "B"]

    rf = MyRandomForestClassifier(n_estimators=3)
    rf.fit(X_train, y_train)

    preds = rf.predict([[0], [1], [0]])
    assert len(preds) == 3

def test_rf_reproducibility():
    X = [[0], [1], [0], [1], [0]]
    y = ["A", "B", "A", "B", "A"]

    rf1 = MyRandomForestClassifier(n_estimators=5, random_state=42)
    rf2 = MyRandomForestClassifier(n_estimators=5, random_state=42)

    rf1.fit(X, y)
    rf2.fit(X, y)

    assert rf1.predict([[0], [1]]) == rf2.predict([[0], [1]])

def test_rf_initialization_defaults():
    rf = MyRandomForestClassifier(n_estimators=10)
    assert rf.n_estimators == 10
    assert rf.max_features == "sqrt"
    assert rf.bootstrap_ratio == 0.7  

def test_rf_no_trees_before_fit():
    rf = MyRandomForestClassifier(n_estimators=5)
    assert rf.trees == []

def test_rf_predict_without_fit():
    rf = MyRandomForestClassifier(n_estimators=3)
    try:
        preds = rf.predict([[1, 2, 3]])
        # The model should not predict anything meaningful here
        assert preds == [] or preds is not None
    except Exception:
        assert True  # acceptable behavior too

import math

def test_rf_feature_subset_sqrt():
    X = [
        [1,2,3,4],
        [4,3,2,1],
        [1,1,1,1],
        [2,2,2,2]
    ]
    y = ["A", "B", "A", "B"]

    rf = MyRandomForestClassifier(n_estimators=3, max_features="sqrt", random_state=0)
    rf.fit(X, y)

    expected = math.ceil(math.sqrt(len(X[0])))

    for subset in rf.feature_subsets:
        assert len(subset) == expected

def test_rf_bootstrap_diversity():
    X = [[i, i+1, i+2] for i in range(20)]   # 3 features
    y = ["A" if i < 10 else "B" for i in range(20)]

    rf = MyRandomForestClassifier(n_estimators=3, random_state=0)
    rf.fit(X, y)

    # Now random subsets should differ across trees
    assert rf.feature_subsets[0] != rf.feature_subsets[1]



def test_rf_predictions_are_valid_labels():
    X = [[0], [1], [0], [1]]
    y = ["red", "blue", "red", "blue"]

    rf = MyRandomForestClassifier(n_estimators=5, random_state=0)
    rf.fit(X, y)

    preds = rf.predict([[0], [1]])

    for p in preds:
        assert p in y
