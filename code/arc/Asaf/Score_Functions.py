import numpy as np
from scipy.stats import rankdata
import arc
import pandas as pd

def class_probability_score(probabilities, labels, u=None):
    scores = 1-probabilities[:, labels]
    return scores


def generelized_inverse_quantile_score(probabilities, labels, u=None):

    if u is None:
        randomized = False
    else:
        randomized = True

    # get number of points
    num_of_points = np.shape(probabilities)[0]

    # sort probabilities from high to low
    sorted = -np.sort(-probabilities)

    # create matrix of cumulative sum of each row
    cumulative_sum = np.cumsum(sorted, axis=1)

    # find ranks of each desired label in each row
    label_ranks = rankdata(-probabilities,method='ordinal', axis=1)[:,labels] - 1

    # compute the scores of each label in each row
    scores = cumulative_sum[np.arange(num_of_points), label_ranks.T].T

    last_label_prob = sorted[np.arange(num_of_points), label_ranks.T].T

    if not randomized:
        scores = scores - last_label_prob
    else:
        scores = scores - np.diag(u) @ last_label_prob

    return scores


def evaluate_predictions(S, X, y, conditional=True):

    # turn one hot vectors into single lables
    y = np.argmax(y, axis=1)

    # get numbers of points
    n = np.shape(X)[0]

    # get point to a matrix of the format nxp
    X = np.vstack([X[i, 0, :, :].flatten() for i in range(n)])

    # Marginal coverage
    marg_coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    if conditional:
        # Estimated conditional coverage (worse-case slab)
        wsc_coverage = arc.coverage.wsc_unbiased(X, y, S, M=100)
    else:
        wsc_coverage = None
    # Size and size conditional on coverage
    size = np.mean([len(S[i]) for i in range(len(y))])
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    size_cover = np.mean([len(S[i]) for i in idx_cover])
    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Size': [size], 'Size cover': [size_cover]})
    return out
