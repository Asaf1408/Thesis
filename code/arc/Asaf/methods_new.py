import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.stats.mstats import mquantiles
#from scipy.special import softmax
from torch.nn.functional import softmax
from numpy.random import default_rng
import sys
from tqdm import tqdm
from scipy.stats import norm
from arc.classification import ProbabilityAccumulator as ProbAccum
from Asaf.Random_Generators import random_in_ball
import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt


# No_Calibration
class No_Calibration:
    def __init__(self, black_box, alpha):

        # store the classifier used
        self.black_box = black_box

        # store alpha
        self.alpha = alpha

    # predict prediction sets
    def predict(self, X):

        # automatically choose device use gpu 0 if it is available o.w. use the cpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # get number of points
        n = X.size()[0]

        # generate random uniform variables for oracle set creation
        rng = default_rng()
        u = rng.uniform(low=0.0, high=1.0, size=n)

        # get classifier predictions on test points
        self.black_box.eval()  # put in evaluation mode
        with torch.no_grad():
            X = X.to(device)
            P_hat = self.black_box(X).to(torch.device('cpu'))

        # transform the output into probabilities vector
        P_hat = softmax(P_hat, dim=1).numpy()

        # generate prediction sets
        grey_box = ProbAccum(P_hat)
        S_hat = grey_box.predict_sets(self.alpha, epsilon=u)

        return S_hat


# Classical conformal prediction
class Non_Conformity_Score_Calibration:
    def __init__(self, X_calib, Y_calib, black_box, alpha, score_func=None):

        # automatically choose device use gpu 0 if it is available o.w. use the cpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # size of the calibration set
        n_calib = X_calib.size()[0]

        # calibrator parameters
        self.black_box = black_box
        self.alpha = alpha
        self.score_func = score_func

        # get classifier predictions on calibration set
        self.black_box.eval()  # put in evaluation mode
        with torch.no_grad():
            X_calib = X_calib.to(device)
            P_hat = self.black_box(X_calib).to(torch.device('cpu'))

        # transform the output into probabilities vector
        P_hat = softmax(P_hat, dim=1).numpy()

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

    # predict prediction sets
    def predict(self, X):

        # automatically choose device use gpu 0 if it is available o.w. use the cpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # get number of points
        n = X.size()[0]

        # get classifier predictions on test set
        self.black_box.eval()  # put in evaluation mode
        with torch.no_grad():
            X = X.to(device)
            P_hat = self.black_box(X).to(torch.device('cpu'))

        # transform the output into probabilities vector
        P_hat = softmax(P_hat, dim=1).numpy()

        # generate random variable for inverse quantile score
        rng = default_rng()
        u = rng.uniform(low=0.0, high=1.0, size=n)

        # compute scores for all points in the test set for all labels
        scores = self.score_func(P_hat, np.arange(self.num_of_classes), u)

        # Generate prediction sets using the threshold from the calibration
        S_hat = [np.where(scores[i, :] <= self.threshold_calibrated)[0] for i in range(n)]

        # return predictions sets
        return S_hat


class Upper_Bound_Score_Calibration:
    def __init__(self, X_calib, Y_calib, black_box, alpha,epsilon=0,score_func=None, ratio=0):

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

        # bring noises to pytorch form
        noises = np.reshape(noises, (self.n_permutations, 1, rows, cols)).astype(np.float32)

        # create container for the scores
        scores = np.zeros(n_calib)

        # estimate upper bound on the score for each point
        for j in range(n_calib):

            # add noise to data point
            noisy_points = X_calib[j, :, :, :] + noises

            # clip to pixel values
            noisy_points[noisy_points < 0] = 0
            noisy_points[noisy_points > 1] = 1

            # get classifier results for the noisy points
            noisy_outputs = self.black_box.predict(noisy_points)

            # transform the output into probabilities vector
            noisy_outputs = softmax(noisy_outputs, dim=1)

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
        P_hat = softmax(P_hat, dim=1)

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
    def __init__(self, X_calib, Y_calib, black_box, alpha, epsilon=0,score_func=None,ratio=0):

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
        P_hat = softmax(P_hat, dim=1)

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

        # bring noises to pytorch form
        noises = np.reshape(noises, (self.n_permutations, 1, rows, cols)).astype(np.float32)

        Scores_Lower_Bounds = np.zeros((n,self.num_of_classes))
        for j in range(n):
            # add noise to data point
            noisy_points = X[j, :, :, :] + noises

            # clip to pixel values
            noisy_points[noisy_points < 0] = 0
            noisy_points[noisy_points > 1] = 1

            # get classifier results for the noisy points
            noisy_outputs = self.black_box.predict(noisy_points)

            # transform the output into probabilities vector
            noisy_outputs = softmax(noisy_outputs, dim=1)

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
    def __init__(self, X_calib, Y_calib, black_box, noises, alpha, epsilon=0, score_func=None, ratio=2, device='cpu'):

        # set device
        self.device = device

        # size of the calibration set
        n_calib = X_calib.size()[0]

        # number of permutations to estimate mean
        self.n_smooth = noises.size()[0]//n_calib

        # calibrator parameters
        self.black_box = black_box
        self.alpha = alpha
        self.score_func = score_func
        self.epsilon = epsilon

        # set standard deviation and mean for smoothing
        self.sigma = ratio * epsilon

        # generate random vectors from the Gaussian distribution
        rng = default_rng()

        # create container for the scores
        scores = np.zeros(n_calib)

        # calculate maximum batch size according to gpu capacity
        batch_size = 1024 // self.n_smooth

        # calculate number of batches
        if n_calib % batch_size != 0:
            num_of_batches = (n_calib // batch_size) + 1
        else:
            num_of_batches = (n_calib // batch_size)

        for j in range(num_of_batches):
            # get inputs and labels of batch
            inputs = X_calib[(j * batch_size):((j + 1) * batch_size)]
            labels = Y_calib[(j * batch_size):((j + 1) * batch_size)]

            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((len(labels) * self.n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, self.n_smooth, 1, 1)).view(tmp.shape).to(self.device)

            # generate random Gaussian noise for the duplicated batch
            noise = noises[(j * (batch_size * self.n_smooth)):((j + 1) * (batch_size * self.n_smooth))].to(self.device)

            # add noise to points
            noisy_points = x_tmp + noise

            # get classifier predictions on noisy points
            self.black_box.eval()  # put in evaluation mode
            with torch.no_grad():
                noisy_outputs = self.black_box(noisy_points).to(torch.device('cpu'))

            # transform the output into probabilities vector
            noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

            # get number of classes
            if j == 0:
                self.num_of_classes = noisy_outputs.shape[1]

            # get smoothed score for each point
            for k in range(len(labels)):
                # generate random variable for inverse quantile score
                u = np.ones(self.n_smooth) * rng.uniform(low=0.0, high=1.0)

                # estimate empirical mean of noisy scores
                tmp_scores = score_func(noisy_outputs[(k*self.n_smooth):((k+1)*self.n_smooth)], labels[k], u)
                scores[(j*batch_size)+k] = np.mean(tmp_scores)

        # Compute threshold
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))
        self.threshold_calibrated = mquantiles(scores, prob=level_adjusted)

        # calculate correction based on the Lipschitz constant
        if self.sigma == 0:
            self.correction = 0
        else:
            self.correction = float(self.epsilon) / float(self.sigma)

        # calculate lowe and upper bounds of correction
        upper_thresh = norm.cdf(norm.ppf(self.threshold_calibrated, loc=0, scale=1)+self.correction, loc=0, scale=1)
        lower_thresh = norm.cdf(norm.ppf(self.threshold_calibrated, loc=0, scale=1)-self.correction, loc=0, scale=1)

        self.upper_quntile = np.size(scores[scores <= upper_thresh])/np.size(scores)
        self.lower_quntile = np.size(scores[scores <= lower_thresh])/np.size(scores)

        # plot histogram with bounds
        # plt.figure()
        # sns.histplot(scores, bins=100)
        # plt.axvline(x=self.threshold_calibrated,color='r')
        # plt.axvline(x=upper_thresh, color='g')
        # plt.axvline(x=lower_thresh, color='g')
        #
        # plt.savefig("Hist.png")
        # exit(1)

    def predict(self, X, noises, to_correct=True):

        # get number of points
        n = X.size()[0]

        # generate random vectors from the Gaussian distribution
        rng = default_rng()

        # create container for the scores
        scores = np.zeros((n, self.num_of_classes))

        # calculate maximum batch size according to gpu capacity
        batch_size = 1024 // self.n_smooth

        # calculate number of batches
        if n % batch_size != 0:
            num_of_batches = (n // batch_size) + 1
        else:
            num_of_batches = (n // batch_size)

        for j in range(num_of_batches):
            # get inputs and labels of batch
            inputs = X[(j * batch_size):((j + 1) * batch_size)]

            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((inputs.size()[0] * self.n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, self.n_smooth, 1, 1)).view(tmp.shape).to(self.device)

            # generate random Gaussian noise for the duplicated batch
            noise = noises[(j * (batch_size * self.n_smooth)):((j + 1) * (batch_size * self.n_smooth))].to(self.device)

            # add noise to points
            noisy_points = x_tmp + noise

            # get classifier predictions on noisy points
            self.black_box.eval()  # put in evaluation mode
            with torch.no_grad():
                noisy_outputs = self.black_box(noisy_points).to(torch.device('cpu'))

            # transform the output into probabilities vector
            noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

            # get smoothed score for each point
            for k in range(inputs.size()[0]):
                # generate random variable for inverse quantile score
                u = np.ones(self.n_smooth) * rng.uniform(low=0.0, high=1.0)

                # estimate smoothed score
                scores[((j*batch_size)+k), :] = np.mean(self.score_func(noisy_outputs[(k*self.n_smooth):((k+1)*self.n_smooth)], np.arange(self.num_of_classes), u), axis=0)

        # Generate prediction sets using the threshold from the calibration
        S_hat = [np.where(norm.ppf(scores[i, :], loc=0, scale=1) <= norm.ppf(self.threshold_calibrated, loc=0, scale=1))[0] for i in range(n)]

        #S_hat = [np.where(noisy_scores[i, :] - correction2 <= self.threshold_calibrated)[0] for i in range(n)]

        if to_correct:
            S_hat_corrected = [np.where(norm.ppf(scores[i, :], loc=0, scale=1) - self.correction <= norm.ppf(self.threshold_calibrated, loc=0, scale=1))[0] for i in range(n)]
        else:
            S_hat_corrected = None

        # return predictions sets
        return S_hat, S_hat_corrected

    def get_quantile_bounds(self):
        return self.lower_quntile, self.upper_quntile