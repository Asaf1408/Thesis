import numpy as np
from sklearn import svm
from sklearn import ensemble
from sklearn import calibration
from sklearn.neural_network import MLPClassifier

import copy


# the oracle classifier, the one that knows the true conditional probabilities of the model
class Oracle:
    # initiate the classifier
    def __init__(self, model):
        # the oracle only needs the true model of the data
        self.model = model

    # the oracle doesn't need any fitting procedure
    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.model.sample_Y(X)

    def predict_proba(self, X):
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        # get true conditional probabilities of the model
        prob = self.model.compute_prob(X)

        # make sure no class has zero probability because of numerical errors
        prob = np.clip(prob, 1e-6, 1.0)

        # normalize the vectors (needed because of the clipping)
        prob = prob / prob.sum(axis=1)[:, None]
        return prob


# support vector classifier
class SVC:
    def __init__(self, calibrate=False,
                 kernel='linear',
                 C=1,
                 clip_proba_factor=0.1,
                 random_state=2020):
        self.model = svm.SVC(kernel=kernel,
                             C=C,
                             probability=True,
                             random_state=random_state)
        self.model_fit = None
        self.calibrated = None
        self.calibrate = calibrate
        self.num_classes = 0
        self.factor = clip_proba_factor

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.model_fit = self.model.fit(X, y)
        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(self.model_fit,
                                                                 method='sigmoid',
                                                                 cv=10)
        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X):
        if (len(X.shape) == 1):
            X = X.reshape((1, X.shape[0]))
        if self.calibrated is None:
            prob = self.model_fit.predict_proba(X)
        else:
            prob = self.calibrated.predict_proba(X)
        prob = np.clip(prob, self.factor / self.num_classes, 1.0)
        prob = prob / prob.sum(axis=1)[:, None]
        return prob


# Random forest classification
class RFC:
    def __init__(self, calibrate=False,
                 n_estimators=1000,
                 criterion="gini",
                 max_depth=None,
                 max_features="auto",
                 min_samples_leaf=1,
                 clip_proba_factor=0.1,
                 random_state=2020):

        self.model = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                     criterion=criterion,
                                                     max_depth=max_depth,
                                                     max_features=max_features,
                                                     min_samples_leaf=min_samples_leaf,
                                                     random_state=random_state)
        self.model_fit = None
        self.calibrated = None
        self.calibrate = calibrate
        self.num_classes = 0
        self.factor = clip_proba_factor

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.model_fit = self.model.fit(X, y)
        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(self.model_fit,
                                                                 method='sigmoid',
                                                                 cv=10)
        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X):
        if (len(X.shape) == 1):
            X = X.reshape((1, X.shape[0]))
        if self.calibrated is None:
            prob = self.model_fit.predict_proba(X)
        else:
            prob = self.calibrated.predict_proba(X)
        prob = np.clip(prob, self.factor / self.num_classes, 1.0)
        prob = prob / prob.sum(axis=1)[:, None]
        return prob


# Neural network classification
class NNet:
    def __init__(self, calibrate=False,
                 hidden_layer_sizes=64,
                 batch_size=128,
                 learning_rate_init=0.01,
                 max_iter=20,
                 clip_proba_factor=0.1,
                 random_state=2020):

        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                   batch_size=batch_size,
                                   learning_rate_init=learning_rate_init,
                                   max_iter=max_iter,
                                   random_state=random_state)
        self.model_fit = None
        self.calibrated = None
        self.calibrate = calibrate
        self.num_classes = 0
        self.factor = clip_proba_factor

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.model_fit = self.model.fit(X, y)
        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(self.model_fit,
                                                                 method='sigmoid',
                                                                 cv=10)
        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X):
        if (len(X.shape) == 1):
            X = X.reshape((1, X.shape[0]))
        if self.calibrated is None:
            prob = self.model_fit.predict_proba(X)
        else:
            prob = self.calibrated.predict_proba(X)
        prob = np.clip(prob, self.factor / self.num_classes, 1.0)
        prob = prob / prob.sum(axis=1)[:, None]
        return prob
