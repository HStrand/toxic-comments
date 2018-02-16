import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

class Stacker(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            print(clf)
            S_test_i = np.zeros((T.shape[0], self.n_folds))
            
            j = 0
            for train_idx, test_idx in folds.split(X):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_holdout)[:,1]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
                print(roc_auc_score(y_holdout, y_pred))
                j += 1

            S_test[:, i] = S_test_i.mean(1)

        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict_proba(S_test)[:,1]
        return y_pred