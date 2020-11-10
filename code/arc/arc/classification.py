import numpy as np


class ProbabilityAccumulator:
    def __init__(self, prob):

        # define num of samples and num of classes
        self.n, self.K = prob.shape

        # sort each probability vector prediction for each sample from highest to lowest
        self.order = np.argsort(-prob, axis=1)

        # create a matrix that each row has the ranks of each label for the sample
        self.ranks = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))

        # calculate cumulative sum of probabilities for each sample from highest to lowest
        self.prob_sort = -np.sort(-prob, axis=1)
        # self.epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)
        self.Z = np.round(self.prob_sort.cumsum(axis=1), 9)

    def predict_sets(self, alpha, randomize=False, epsilon=None):

        # check which label is the last one to add in order to pass the desired coverage level for each sample
        L = np.argmax(self.Z >= 1.0 - alpha, axis=1).flatten()

        # use a generated uniform variable to choose whether to take the last label or not
        if randomize:
            epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)

        if epsilon is not None:
            # check by how much we exceed the desired probability in each sample
            Z_excess = np.array([self.Z[i, L[i]] for i in range(self.n)]) - (1.0 - alpha)

            # calculate probability to remove the last label
            p_remove = Z_excess / np.array([self.prob_sort[i, L[i]] for i in range(self.n)])

            # if generated uniform variable is smaller than the probability, remove the last label
            remove = epsilon <= p_remove
            for i in np.where(remove)[0]:
                L[i] = L[i] - 1

        # Return prediction set for each sample
        S = [self.order[i, np.arange(0, L[i] + 1)] for i in range(self.n)]
        return (S)

    def calibrate_scores(self, Y, epsilon=None):

        # get true labels of calibration sets
        Y = np.atleast_1d(Y)

        # get size of calibration set
        n2 = len(Y)

        # check what is the rank of the true labels in the prediction
        ranks = np.array([self.ranks[i, Y[i]] for i in range(n2)])

        # calculate how much probability is accumulated until the true label is inside the set
        prob_cum = np.array([self.Z[i, ranks[i]] for i in range(n2)])
        prob = np.array([self.prob_sort[i, ranks[i]] for i in range(n2)])
        alpha_max = 1.0 - prob_cum

        if epsilon is not None:
            alpha_max += np.multiply(prob, epsilon)
        else:
            alpha_max += prob
        alpha_max = np.minimum(alpha_max, 1)
        return alpha_max
