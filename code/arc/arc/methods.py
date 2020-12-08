import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.stats.mstats import mquantiles
import sys
from tqdm import tqdm

from arc.classification import ProbabilityAccumulator as ProbAccum


# Generate "num_points" random points in "dimension" that have uniform
# probability over the unit ball scaled by "radius" (length of points
# are in range [0, "radius"]).
def random_in_ball(num_points, dimension, radius=1, norm="l2"):
    from numpy import random, linalg

    if norm == "l2":
        # First generate random directions by normalizing the length of a
        # vector of random-normal values (these distribute evenly on ball).
        random_directions = random.normal(size=(dimension,num_points))
        random_directions /= linalg.norm(random_directions, axis=0)
        # Second generate a random radius with probability proportional to
        # the surface area of a ball with a given radius.
        random_radii = random.random(num_points) ** (1/dimension)
        # Return the list of random (direction & length) points.
        res = radius * (random_directions * random_radii).T

    elif norm == "infinity":
        res = random.uniform(low=-radius,high=radius,size=(num_points,dimension))

    return res

class CVPlus:
    def __init__(self, X, Y, black_box, alpha, n_folds=10, random_state=2020, verbose=False):
        X = np.array(X)
        Y = np.array(Y)
        self.black_box = black_box
        self.n = X.shape[0]
        self.classes = np.unique(Y)
        self.n_classes = len(self.classes)
        self.n_folds = n_folds
        self.cv = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        self.alpha = alpha
        self.verbose = verbose
        
        # Fit prediction rules on leave-one out datasets
        self.mu_LOO = [ black_box.fit(X[train_index], Y[train_index]) for train_index, _ in self.cv.split(X) ]

        # Accumulate probabilities for the original data with the grey boxes
        test_indices = [test_index for _, test_index in self.cv.split(X)]
        self.test_indices = test_indices
        self.folds = [[]]*self.n
        for k in range(self.n_folds):
            for i in test_indices[k]:
                self.folds[i] = k
        self.grey_boxes = [[]]*self.n_folds
        if self.verbose:
            print("Training black boxes on {} samples with {}-fold cross-validation:".
                  format(self.n, self.n_folds), file=sys.stderr)
            sys.stderr.flush()
            for k in tqdm(range(self.n_folds), ascii=True, disable=True):
                self.grey_boxes[k] = ProbAccum(self.mu_LOO[k].predict_proba(X[test_indices[k]]))
        else:
            for k in range(self.n_folds):
                self.grey_boxes[k] = ProbAccum(self.mu_LOO[k].predict_proba(X[test_indices[k]]))
               
        # Compute scores using real labels
        epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)
        self.alpha_max = np.zeros((self.n, 1))
        if self.verbose:
            print("Computing scores for {} samples:". format(self.n), file=sys.stderr)
            sys.stderr.flush()
            for k in tqdm(range(self.n_folds), ascii=True, disable=True):
                idx = test_indices[k]
                self.alpha_max[idx,0] = self.grey_boxes[k].calibrate_scores(Y[idx], epsilon=epsilon[idx])
        else:
            for k in range(self.n_folds):
                idx = test_indices[k]
                self.alpha_max[idx,0] = self.grey_boxes[k].calibrate_scores(Y[idx], epsilon=epsilon[idx])
            
    def predict(self, X):
        n = X.shape[0]
        S = [[]]*n
        n_classes = len(self.classes)

        epsilon = np.random.uniform(low=0.0, high=1.0, size=n)
        prop_smaller = np.zeros((n,n_classes))

        if self.verbose:
            print("Computing predictive sets for {} samples:". format(n), file=sys.stderr)
            sys.stderr.flush()
            for fold in tqdm(range(self.n_folds), ascii=True, disable=True):
                gb = ProbAccum(self.mu_LOO[fold].predict_proba(X))
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    for i in self.test_indices[fold]:
                        prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])
        else:
            for fold in range(self.n_folds):
                gb = ProbAccum(self.mu_LOO[fold].predict_proba(X))
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    for i in self.test_indices[fold]:
                        prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])

        for k in range(n_classes):
            prop_smaller[:,k] /= float(self.n)
                
        level_adjusted = (1.0-self.alpha)*(1.0+1.0/float(self.n))
        S = [ np.where(prop_smaller[i,:] < level_adjusted)[0] for i in range(n) ]
        return S

class JackknifePlus:
    def __init__(self, X, Y, black_box, alpha, random_state=2020, verbose=False):
        self.black_box = black_box
        self.n = X.shape[0]
        self.classes = np.unique(Y)
        self.alpha = alpha
        self.verbose = verbose

        # Fit prediction rules on leave-one out datasets
        self.mu_LOO = [[]] * self.n
        if self.verbose:
            print("Training black boxes on {} samples with the Jacknife+:". format(self.n), file=sys.stderr)
            sys.stderr.flush()
            for i in range(self.n):
                print("{} of {}...".format(i+1, self.n), file=sys.stderr)
                sys.stderr.flush()
                self.mu_LOO[i] = black_box.fit(np.delete(X,i,0),np.delete(Y,i))
        else:
            for i in range(self.n):
                self.mu_LOO[i] = black_box.fit(np.delete(X,i,0),np.delete(Y,i))

        # Accumulate probabilities for the original data with the grey boxes
        self.grey_boxes = [ ProbAccum(self.mu_LOO[i].predict_proba(X[i])) for i in range(self.n) ]

        # Compute scores using real labels
        epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)
        
        self.alpha_max = np.zeros((self.n, 1))    
        if self.verbose:
            print("Computing scores for {} samples:". format(self.n), file=sys.stderr)
            sys.stderr.flush()
            for i in range(self.n):
                print("{} of {}...".format(i+1, self.n), file=sys.stderr)
                sys.stderr.flush()
                self.alpha_max[i,0] = self.grey_boxes[i].calibrate_scores(Y[i], epsilon=epsilon[i])
        else:
            for i in range(self.n):
                self.alpha_max[i,0] = self.grey_boxes[i].calibrate_scores(Y[i], epsilon=epsilon[i])
                
    def predict(self, X):
        n = X.shape[0]
        S = [[]]*n
        n_classes = len(self.classes)

        epsilon = np.random.uniform(low=0.0, high=1.0, size=n)
        prop_smaller = np.zeros((n,n_classes))
        
        if self.verbose:
            print("Computing predictive sets for {} samples:". format(n), file=sys.stderr)
            sys.stderr.flush()
            for i in range(self.n):
                print("{} of {}...".format(i+1, self.n), file=sys.stderr)
                sys.stderr.flush()
                gb = ProbAccum(self.mu_LOO[i].predict_proba(X))
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])
        else:
            for i in range(self.n):
                gb = ProbAccum(self.mu_LOO[i].predict_proba(X))
                for k in range(n_classes):
                    y_lab = [self.classes[k]] * n
                    alpha_max_new = gb.calibrate_scores(y_lab, epsilon=epsilon)
                    prop_smaller[:,k] += (alpha_max_new < self.alpha_max[i])
                
        for k in range(n_classes):
            prop_smaller[:,k] /= float(self.n)
        level_adjusted = (1.0-self.alpha)*(1.0+1.0/float(self.n))
        S = [ np.where(prop_smaller[i,:] < level_adjusted)[0] for i in range(n) ]
        return S


# Classical conformal prediction
class SplitConformal:
    def __init__(self, X, Y, black_box, alpha, random_state=2020, verbose=False):
        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state)

        # size of the calibration set
        n2 = X_calib.shape[0]

        # the classifier used
        self.black_box = black_box

        # Fit model on training set
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data

        # get model  conditional probabilities estimation on calibration set
        p_hat_calib = self.black_box.predict_proba(X_calib)
        grey_box = ProbAccum(p_hat_calib)

        epsilon = np.random.uniform(low=0.0, high=1.0, size=n2)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store calibrate level
        self.alpha_calibrated = alpha - alpha_correction

    # predict prediction sets
    def predict(self, X):
        n = X.shape[0]
        epsilon = np.random.uniform(low=0.0, high=1.0, size=n)
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon)
        return S_hat

# Classical conformal prediction
class No_Calibration:
    def __init__(self, X, Y, black_box, alpha, random_state=2020, verbose=False):

        # the classifier used
        self.black_box = black_box

        # Fit model on training set
        self.black_box.fit(X, Y)

        # store alpha
        self.alpha = alpha

    # predict prediction sets
    def predict(self, X):
        n = X.shape[0]
        epsilon = np.random.uniform(low=0.0, high=1.0, size=n)
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat = grey_box.predict_sets(self.alpha, epsilon=epsilon)
        return S_hat


class Split_Lower_Bound_Score:
    def __init__(self, X, Y, black_box, alpha, random_state=2020, verbose=False,epsilon=0):
        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=random_state)

        # size of the calibration set
        n_calib = X_calib.shape[0]

        # dimension of data
        p = X_calib.shape[1]

        # number of permutations to compute lower and upper bounds
        n_permutations = 100

        self.black_box = black_box
        self.alpha = alpha

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # get number of classes
        tmp = self.black_box.predict_proba(X_calib[0,:])
        num_of_classes = tmp.shape[1]

        # generate noise perturbations distributed uniformly inside an unit lp ball with radius epsilon
        noises = random_in_ball(n_permutations,dimension=p, radius=epsilon, norm="l2")

        # create containers for lower and upper bounds for each point
        Lower_Bounds = np.zeros((n_calib, num_of_classes))
        Upper_Bounds = np.zeros((n_calib, num_of_classes))

        # estimate bounds for input of classifier for each noisy point
        for j in range(n_calib):

            # add noise to data point
            noisy_points = X_calib[j,:] + noises

            # get classifier results for the noisy points
            noisy_outputs = self.black_box.predict_proba(noisy_points)

            # estimate empirical lower and upper bounds for the point output under this noise
            Lower_Bounds[j,:] = np.min(noisy_outputs,axis=0)
            Upper_Bounds[j,:] = np.max(noisy_outputs,axis=0)

        # compute score for each point in the calibration set
        scores = np.array([Lower_Bounds[i,Y_calib[i]] for i in range(n_calib)])

        # Compute threshold
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
        self.threshold_calibrated = mquantiles(scores, prob=1.0 - level_adjusted)

    def predict(self, X):
        # get number of points
        n = X.shape[0]

        # get classifier output for the points
        p_hat = self.black_box.predict_proba(X)

        # Generate prediction sets using the threshold from the calibration
        S_hat = [np.where(p_hat[i, :] >= self.threshold_calibrated)[0] for i in range(n)]

        # return predictions sets
        return S_hat
