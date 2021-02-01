import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.stats.mstats import mquantiles
from scipy.special import softmax
from numpy.random import default_rng
import sys
from tqdm import tqdm
from scipy.stats import norm
from arc.classification import ProbabilityAccumulator as ProbAccum
from Asaf.Random_Generators import random_in_ball
import torch

# No_Calibration
class No_Calibration:
    def __init__(self, black_box, alpha):

        # store the classifier used
        self.black_box = black_box

        # store alpha
        self.alpha = alpha

    # predict prediction sets
    def predict(self, X):
        # get number of points
        n = np.shape(X)[0]

        # generate random uniform variables for oracle set creation
        rng = default_rng()
        u = rng.uniform(low=0.0, high=1.0,size=n)

        # get classifier predictions on test points
        P_hat = self.black_box.predict(X)

        # transform the output into probabilities vector
        P_hat = softmax(P_hat, axis=1)

        # generate prediction sets
        grey_box = ProbAccum(P_hat)
        S_hat = grey_box.predict_sets(self.alpha, epsilon=u)

        return S_hat


# Classical conformal prediction
class Non_Conformity_Score_Calibration:
    def __init__(self, X_calib, Y_calib, black_box, alpha, score_func=None):

        # size of the calibration set
        n_calib = np.shape(X_calib)[0]

        # turn one hot vectors into single lables
        Y_calib = np.argmax(Y_calib, axis=1)

        # calibrator parameters
        self.black_box = black_box
        self.alpha = alpha
        self.score_func = score_func

        # get classifier predictions on calibration set
        P_hat = self.black_box.predict(X_calib)

        # transform the output into probabilities vector
        P_hat = softmax(P_hat, axis=1)

        # get number of classes
        self.num_of_classes = P_hat.shape[1]

        # generate random variable for inverse quantile score
        rng = default_rng()
        u = rng.uniform(low=0.0, high=1.0,size=n_calib)

        # compute scores for all points in the calibration set
        scores = self.score_func(P_hat,np.arange(self.num_of_classes),u)[np.arange(n_calib), Y_calib.T].T

        # Compute threshold
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
        self.threshold_calibrated = mquantiles(scores, prob=level_adjusted)

    # predict prediction sets
    def predict(self, X):
        # get number of points
        n = np.shape(X)[0]

        # get classifier predictions on test set
        P_hat = self.black_box.predict(X)

        # transform the output into probabilities vector
        P_hat = softmax(P_hat, axis=1)

        # generate random variable for inverse quantile score
        rng = default_rng()
        u = rng.uniform(low=0.0, high=1.0,size=n)

        # compute scores for all points in the test set for all labels
        scores = self.score_func(P_hat, np.arange(self.num_of_classes), u)

        # Generate prediction sets using the threshold from the calibration
        S_hat = [np.where(scores[i, :] <= self.threshold_calibrated)[0] for i in range(n)]

        # return predictions sets
        return S_hat


class Upper_Bound_Score_Calibration:
    def __init__(self, X_calib, Y_calib, black_box, alpha,epsilon=0,score_func=None):

        # size of the calibration set
        n_calib = X_calib.shape[0]

        # turn one hot vectors into single lables
        Y_calib = np.argmax(Y_calib, axis=1)

        # number of permutations to compute score upper bound
        self.n_permutations = 1000

        # calibrator parameters
        self.black_box = black_box
        self.alpha = alpha
        self.score_func = score_func

        # get number of classes
        tmp = self.black_box.predict(X_calib[0:1,:,:,:])
        self.num_of_classes = tmp.shape[1]

        # get dimension of data
        rows = np.shape(X_calib)[2]
        cols = np.shape(X_calib)[3]
        p = rows * cols

        # generate noise perturbations distributed uniformly inside an unit lp ball with radius epsilon
        noises = random_in_ball(self.n_permutations,dimension=p, radius=epsilon, norm="l2")

        # clip to pixel values
        noises[noises < 0] = 0
        noises[noises > 1] = 1

        # bring noises to pytorch form
        noises = np.reshape(noises, (self.n_permutations, 1, rows, cols)).astype(np.float32)

        # create container for the scores
        scores = np.zeros(n_calib)

        # estimate upper bound on the score for each point
        for j in range(n_calib):

            # add noise to data point
            noisy_points = X_calib[j, :, :, :] + noises

            # get classifier results for the noisy points
            noisy_outputs = self.black_box.predict(noisy_points)

            # transform the output into probabilities vector
            noisy_outputs = softmax(noisy_outputs, axis=1)

            # generate random variable for inverse quantile score
            rng = default_rng()
            u = np.ones(self.n_permutations) * rng.uniform(low=0.0, high=1.0)

            # estimate empirical upper bound for the point score under this noise
            scores[j] = np.max(score_func(noisy_outputs, Y_calib[j], u))

        # Compute threshold
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
        self.threshold_calibrated = mquantiles(scores, prob=level_adjusted)

    def predict(self, X):
        # get number of points
        n = X.shape[0]

        # get classifier predictions on test set
        P_hat = self.black_box.predict(X)

        # transform the output into probabilities vector
        P_hat = softmax(P_hat, axis=1)

        # generate random variable for inverse quantile score
        rng = default_rng()
        u = rng.uniform(low=0.0, high=1.0,size=n)

        # compute scores for all points in the test set for all labels
        scores = self.score_func(P_hat, np.arange(self.num_of_classes), u)

        # Generate prediction sets using the threshold from the calibration
        S_hat = [np.where(scores[i, :] <= self.threshold_calibrated)[0] for i in range(n)]

        # return predictions sets
        return S_hat


class Test_Score_Lower_Bound_Calibration:
    def __init__(self, X_calib, Y_calib, black_box, alpha, epsilon=0,score_func=None):

        # size of the calibration set
        n_calib = X_calib.shape[0]

        # turn one hot vectors into single lables
        Y_calib = np.argmax(Y_calib, axis=1)

        # calibrator parameters
        self.black_box = black_box
        self.alpha = alpha
        self.score_func = score_func
        self.epsilon = epsilon

        # get classifier predictions on calibration set
        P_hat = self.black_box.predict(X_calib)

        # transform the output into probabilities vector
        P_hat = softmax(P_hat, axis=1)

        # get number of classes
        self.num_of_classes = P_hat.shape[1]

        # generate random variable for inverse quantile score
        rng = default_rng()
        u = rng.uniform(low=0.0, high=1.0, size=n_calib)

        # compute scores for all points in the calibration set
        scores = self.score_func(P_hat, np.arange(self.num_of_classes), u)[np.arange(n_calib), Y_calib.T].T

        # Compute threshold
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
        self.threshold_calibrated = mquantiles(scores, prob=level_adjusted)

    def predict(self, X):
        # get number of points
        n = X.shape[0]

        # get dimension of data
        rows = np.shape(X)[2]
        cols = np.shape(X)[3]
        p = rows * cols

        # number of permutations to compute lower and upper bounds
        self.n_permutations = 100

        # generate noise perturbations distributed uniformly inside an unit lp ball with radius epsilon
        noises = random_in_ball(self.n_permutations,dimension=p, radius=self.epsilon, norm="l2")

        # clip to pixel values
        noises[noises < 0] = 0
        noises[noises > 1] = 1

        # bring noises to pytorch form
        noises = np.reshape(noises, (self.n_permutations, 1, rows, cols)).astype(np.float32)

        Scores_Lower_Bounds = np.zeros((n,self.num_of_classes))
        for j in range(n):
            # add noise to data point
            noisy_points = X[j, :, :, :] + noises

            # get classifier results for the noisy points
            noisy_outputs = self.black_box.predict(noisy_points)

            # transform the output into probabilities vector
            noisy_outputs = softmax(noisy_outputs, axis=1)

            # generate random variable for inverse quantile score
            rng = default_rng()
            u = np.ones(self.n_permutations) * rng.uniform(low=0.0, high=1.0)

            # compute score of all labels
            Scores_Lower_Bounds[j, :] = np.min(self.score_func(noisy_outputs,np.arange(self.num_of_classes),u),axis=0)

        # Generate prediction sets using the threshold from the calibration
        S_hat = [np.where(Scores_Lower_Bounds[i,:] <= self.threshold_calibrated)[0] for i in range(n)]

        # return predictions sets
        return S_hat


class Smoothed_Score_Calibration:
    def __init__(self, X_calib, Y_calib, black_box, alpha, epsilon=0, score_func=None):

        # size of the calibration set
        n_calib = np.shape(X_calib)[0]

        # turn one hot vectors into single labels
        Y_calib = np.argmax(Y_calib, axis=1)

        # get dimension of data
        rows = np.shape(X_calib)[2]
        cols = np.shape(X_calib)[3]
        channels = np.shape(X_calib)[1]
        p = rows * cols

        # number of permutations to estimate mean
        self.n_permutations = 1000

        # calibrator parameters
        self.black_box = black_box
        self.alpha = alpha
        self.score_func = score_func
        self.epsilon = epsilon

        # get number of classes
        tmp = self.black_box.predict(X_calib[0:1, :, :, :])
        self.num_of_classes = tmp.shape[1]

        # set standard deviation and mean for smoothing
        self.sigma = (10) * epsilon
        self.mean = 0

        # generate random vectors from the Gaussian distribution
        rng = default_rng()
        noises = rng.normal(0, self.sigma, (self.n_permutations, channels, rows, cols)).astype(np.float32)

        # clip to pixel values
        noises[noises < 0] = 0
        noises[noises > 1] = 1

        # create container for the scores
        scores = np.zeros(n_calib)

        # estimate mean over all noise added points
        for j in range(n_calib):

            # add noise to data point
            noisy_points = X_calib[j, :, :, :] + noises

            # get classifier results for the noisy points
            noisy_outputs = self.black_box.predict(noisy_points)

            # transform the output into probabilities vector
            noisy_outputs = softmax(noisy_outputs, axis=1)

            # generate random variable for inverse quantile score
            u = np.ones(self.n_permutations) * rng.uniform(low=0.0, high=1.0)

            # estimate empirical mean of noisy scores
            scores[j] = np.mean(score_func(noisy_outputs,Y_calib[j],u))

        # Compute threshold
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
        self.threshold_calibrated = mquantiles(scores, prob=level_adjusted)

    def predict(self, X):
        # get number of points
        n = np.shape(X)[0]

        # get dimension of data
        rows = np.shape(X)[2]
        cols = np.shape(X)[3]
        channels = np.shape(X)[1]
        p = rows * cols

        # generate random vectors from the Gaussian distribution
        rng = default_rng()
        noises = rng.normal(0, self.sigma, (self.n_permutations, channels, rows, cols)).astype(np.float32)

        # clip to pixel values
        noises[noises < 0] = 0
        noises[noises > 1] = 1

        # create container for the noisy scores
        noisy_scores = np.zeros((n,self.num_of_classes))
        for j in range(n):
            # add noise to data point
            noisy_points = X[j, :, :, :] + noises

            # get classifier results for the noisy points
            noisy_outputs = self.black_box.predict(noisy_points)

            # transform the output into probabilities vector
            noisy_outputs = softmax(noisy_outputs, axis=1)

            # generate random variable for inverse quantile score
            u = np.ones(self.n_permutations) * rng.uniform(low=0.0, high=1.0)

            # compute score of all labels
            noisy_scores[j,:] = np.mean(self.score_func(noisy_outputs,np.arange(self.num_of_classes),u),axis=0)

        # correction based on the Lipschitz constant
        if self.sigma == 0:
            correction1 = 0
            correction2 = 0
        else:
            correction1 = self.epsilon / self.sigma
            correction2 = (self.epsilon / self.sigma)*np.sqrt(2/np.pi)

        # Generate prediction sets using the threshold from the calibration
        S_hat = [np.where(norm.ppf(noisy_scores[i,:],loc=0,scale=1) - correction1 <= norm.ppf(self.threshold_calibrated,loc=0,scale=1))[0] for i in range(n)]
        #S_hat = [np.where(noisy_scores[i, :] - correction2 <= self.threshold_calibrated)[0] for i in range(n)]
        # return predictions sets
        return S_hat