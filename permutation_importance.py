import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
np.random.seed(0)

def count_fp(y_true, y_pred):
    
    cm = confusion_matrix(y_true, y_pred)

    return cm[0, 1]

def permutation_feature_importance(X, y, X_test, y_test):

    """
    Calculates Permutation Feature Importance based on decision tree classifier
    Differs slightly from sklearn implementation
    
    """

    clf = DecisionTreeClassifier(random_state=np.random.seed(0))
    clf.fit(X,y)
    y_pred = clf.predict(X_test)
    initial_acc = accuracy_score(y_test, y_pred)
    intial_fp = count_fp(y_test, y_pred)

    imp_scores = []
    for col in X.columns:
        X_perm = X.copy()
        np.random.seed(0)
        X_perm[col] = np.random.permutation(X_perm[col])
        # train new clf
        perm_clf = DecisionTreeClassifier(random_state=np.random.seed(0))
        perm_clf.fit(X_perm, y)
        permuted_score = accuracy_score(y_test, perm_clf.predict(X_test))
        imp_score = permuted_score - initial_acc
        imp_scores.append((col, imp_score))

    return intial_fp, sorted(imp_scores, key=lambda x: abs(x[1]), reverse=True)


def run_clf(data_train, data_test, n):

    results = {}

    train_tar = data_train[["target"]]
    train_feat = data_train.drop(columns=["target"])

    test_tar = data_test[["target"]]
    test_feat = data_test.drop(columns=["target"])
    
    base_fp, feat_imps = permutation_feature_importance(train_feat, train_tar,
    test_feat, test_tar)

    results["fp"] = base_fp
    results["most_important"] = feat_imps

    final_feat = [x[0] for x in feat_imps[:n]]

    clf = DecisionTreeClassifier(random_state=np.random.seed(0))
    clf.fit(train_feat[final_feat], train_tar)
    y_test_pred = clf.predict(test_feat[final_feat])
    fp_most_important = count_fp(test_tar, y_test_pred)

    results["fp_most_important"] = fp_most_important

    return results


train_data = pd.read_csv("credit_data/train.csv")
test_data = pd.read_csv("credit_data/test.csv")
print(train_data.shape, test_data.shape)
print(run_clf(train_data, test_data, 24))