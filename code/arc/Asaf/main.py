import numpy as np
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import arc
import multiprocessing as mp
from arc.methods import random_in_ball
from arc.methods import random_in_sphere
from experiments_real_data.datasets import GetDataset
import Asaf.Score_Functions as scores
#import art
#from art.attacks.evasion import FastGradientMethod
#from art.estimators.classification import SklearnClassifier
#from art.attacks.evasion import universal_perturbation
#from art.attacks.evasion import AutoAttack
import sys

import os

np.random.seed(2020)


def evaluate_predictions(S, X, y, conditional=True):
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

def run_experiment_pool(args):
    # get arguments of this check
    black_boxes, methods, X_calib, Y_calib, X_test, Y_test, box_name, method_name, noise_norm, alpha, random_state = args

    # get dimension of data
    p = len(X_calib[0, :])

    # get size of test and calibration sets
    n_test = len(Y_test)
    n_calib = len(Y_calib)

    # get current classifier
    black_box = black_boxes[box_name]

    # add noise to test set in a epsilon ball
    X_test = X_test + random_in_ball(n_test, p, radius=noise_norm, norm="l2")

    #clf = SklearnClassifier(model=black_box)
    #y_train_one_hot = np.zeros((y_train.size, y_train.max() + 1))
    #y_train_one_hot[np.arange(y_train.size), y_train] = 1
    #clf.fit(X_train,y_train_one_hot)
    #adv_crafter = FastGradientMethod(estimator=clf, norm=2, eps=noise_norm)
    #X_test = adv_crafter.generate(x=X_test)

    # Calibrate model with calibration method
    if method_name == "HCC_LB" or method_name == "HCC_UB" or method_name == "HCC_Smooth":
        method = methods[method_name](X_calib, Y_calib, black_box, alpha, epsilon=noise_norm, score_func=scores.class_probability_score)
    elif method_name == "SC_LB" or method_name == "SC_UB" or method_name == "SC_Smooth":
        method = methods[method_name](X_calib, Y_calib, black_box, alpha, epsilon=noise_norm, score_func=scores.generelized_inverse_quantile_score)
    else:
        method = methods[method_name](X_calib, Y_calib, black_box, alpha, random_state=random_state, verbose=False)

    # Form prediction sets for test points
    S = method.predict(X_test)

    # Evaluate results
    res = evaluate_predictions(S, X_test, Y_test)

    res['Method'] = method_name
    res['Black box'] = box_name
    res['Nominal'] = 1 - alpha
    res['n_test'] = n_test
    res['n_calib'] = n_calib
    res['noise_norm'] = noise_norm

    return res


def run_experiment(X, Y, methods, black_boxes, noises_norms, condition_on, alpha=0.1, experiment=0, random_state=2020):
    # Set random seed
    np.random.seed(random_state)

    # Split test data into calibration and test
    X_test, X_calib, Y_test, Y_calib = train_test_split(X, Y, test_size=0.8, random_state=random_state)

    # create parameter list for each run
    parametrs = []
    for box_name in black_boxes:
        for method_name in methods:
            for noise_norm in noises_norms:
                parametrs.append((black_boxes, methods, X_calib, Y_calib, X_test, Y_test, box_name, method_name,
                                  noise_norm, alpha, random_state))

    # check for number of cores
    workers = mp.cpu_count()
    print("number of workers: " + str(workers))

    # create pool with number of cores
    pool = mp.Pool(workers)

    # create dataframe for storing results
    results = pd.DataFrame()
    for res in pool.map(run_experiment_pool, parametrs):

        # Add information about this experiment
        res['Experiment'] = experiment

        # Add results to the list
        results = results.append(res)

    return results


def run_experiment2(args):

    # get arguments of this check
    X, Y, methods, black_boxes, noises_norms, condition_on, alpha, experiment, random_state = args

    # Set random seed
    np.random.seed(random_state)

    # Split test data into calibration and test
    X_test, X_calib, Y_test, Y_calib = train_test_split(X, Y, test_size=0.8, random_state=random_state)

    # get dimension of data
    p = len(X_calib[0, :])

    # get size of test and calibration sets
    n_test = len(Y_test)
    n_calib = len(Y_calib)

    # create dataframe for storing results
    results = pd.DataFrame()

    for box_name in black_boxes:
        for method_name in methods:
            for noise_norm in noises_norms:

                # get current classifier
                black_box = black_boxes[box_name]

                # add noise to test set in a epsilon ball
                X_test = X_test + random_in_ball(n_test, p, radius=noise_norm, norm="l2")

                # Calibrate model with calibration method
                if method_name == "HCC_LB" or method_name == "HCC_UB" or method_name == "HCC_Smooth":
                    method = methods[method_name](X_calib, Y_calib, black_box, alpha, epsilon=noise_norm,
                                                  score_func=scores.class_probability_score)
                elif method_name == "SC_LB" or method_name == "SC_UB" or method_name == "SC_Smooth":
                    method = methods[method_name](X_calib, Y_calib, black_box, alpha, epsilon=noise_norm,
                                                  score_func=scores.generelized_inverse_quantile_score)
                else:
                    method = methods[method_name](X_calib, Y_calib, black_box, alpha, random_state=random_state,
                                                  verbose=False)

                # Form prediction sets for test points
                S = method.predict(X_test)

                # Evaluate results
                res = evaluate_predictions(S, X_test, Y_test)

                res['Method'] = method_name
                res['Black box'] = box_name
                res['Nominal'] = 1 - alpha
                res['n_test'] = n_test
                res['n_calib'] = n_calib
                res['noise_norm'] = noise_norm
                res['Experiment'] = experiment

                # Add results to the list
                results = results.append(res)

    # print how many experiments have finished
    print("experiment "+str(experiment)+" finished")
    return results


if __name__ == '__main__':
    import warnings

    # close all figures from previous run
    plt.close('all')

    data_type = "real"          # "generated or "real" data sets
    dataset_name = "mnist"      # dataset name if real data is used
    model_num = 1               # define model num for generating data if generated data is choosed
    alpha = 0.1                 # define desired conditional coverage (1-alpha)
    n_train = 1000              # define number of samples in the training set
    n_test = 5000               # define number of samples in the test set
    n_experiments = 10          # define Number of independent experiments
    condition_on = [0]          # define features to condition on

    # define vector of additive noises radius
    epsilons = [0, 1e-1, 5 * (1e-1), 1e0, 5 * (1e0)]
    #epsilons = [0,0.5,1,1.5,2,2.5,3]
    epsilons = [0.1]

    # Total number of samples
    n = n_train + n_test

    if data_type == "real":
        data_model = dataset_name

        # load dataset
        dataset_base_path = os.getcwd()
        X, Y = GetDataset(data_model, dataset_base_path)
        Y = Y.astype(np.long)

        # reduce number of data points
        X = X[0:n, :]
        Y = Y[0:n]

    else:
        # Define data model
        if model_num == 1:
            K = 10  # number of classes
            p = 10  # dimension of data
            data_model = arc.models.Model_Ex1(K, p)  # create model
        else:
            K = 4  # number of classes
            p = 5  # dimension of data
            data_model = arc.models.Model_Ex2(K, p)  # create model

        # Generate data with labels
        X = data_model.sample_X(n)
        Y = data_model.sample_Y(X)

    # Split data into train/test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=n_test)

    # List of calibration methods to be compared
    methods = {
        #'None': arc.methods.No_Calibration,
        #'SC': arc.methods.SplitConformal,
        # 'CV+': arc.methods.CVPlus,
        # 'JK+': arc.methods.JackknifePlus,
         #'HCC': arc.others.SplitConformalHomogeneous,
        # 'CQC': arc.others.CQC
         #'HCC_LB': arc.methods.Split_Score_Lower_Bound,
         #'SC_LB': arc.methods.Split_Score_Lower_Bound,
         #'HCC_UB': arc.methods.Split_Score_Upper_Bound,
         #'SC_UB': arc.methods.Split_Score_Upper_Bound,
        'SC_Smooth': arc.methods.Split_Smooth_Score
        #'HCC_Smooth': arc.methods.Split_Smooth_Score
    }

    # List of black boxes to be compared
    black_boxes = {}
    #if data_type == "generated":
    #    black_boxes.update({'Oracle': arc.black_boxes.Oracle(data_model)})

    black_boxes.update({
        'SVC': arc.black_boxes.SVC(clip_proba_factor=1e-5, random_state=2020)
        #'RFC': arc.black_boxes.RFC(clip_proba_factor=1e-5, n_estimators=1000, max_depth=5, max_features=None,random_state=2020)
    })

    # fit all models on train data
    for box_name in black_boxes:
        black_boxes[box_name].fit(X_train, Y_train)

    # create parameter list for each run
    parametrs = []
    for experiment in range(n_experiments):
        random_state = 2020 + experiment
        parametrs.append((X_test, Y_test, methods, black_boxes, epsilons, condition_on, alpha, experiment, random_state))

    # check for number of cores
    workers = mp.cpu_count()
    print("number of workers: " + str(workers))

    # create pool with number of cores
    pool = mp.Pool(workers)

    # create dataframe for storing the results
    results = pd.DataFrame()

    start = time.time()
#    # run experiments with all the calibration methods and black boxes
#    for res in pool.map(run_experiment2, parametrs):
#        # Add results to the list
#        results = results.append(res)


    # run experiments with all the calibration methods and black boxes
    for experiment in tqdm(range(n_experiments)):
        print("Experiment " + str(experiment) + ":")
        # Random state for this experiment
        random_state = 2020 + experiment

        res = run_experiment(X_test, Y_test, methods, black_boxes, epsilons, condition_on,
                             alpha=alpha, experiment=experiment, random_state=random_state)
        results = results.append(res)

    end = time.time()
    print(end - start)

    # compute SNR
    # get dimension of data
    p = len(X[0, :])
    P_x = np.mean(np.sum(X ** 2, axis=1))
    SNR = np.zeros(np.size(epsilons))
    for i in range(np.size(epsilons)):
        noises = random_in_ball(5000, p, radius=epsilons[i], norm="l2")
        P_noise = np.mean(np.sum(noises ** 2, axis=1))
        SNR[i] = round(10 * np.log10(P_x/P_noise),1)

    # plot marginal coverage results
    ax = sns.catplot(x="Black box", y="Coverage",
                     hue="Method", col="noise_norm",
                     data=results, kind="box",
                     height=4, aspect=.7)
    # ax = sns.boxplot(y="Coverage", x="Black box", hue="Method", data=results)
    for i, graph in enumerate(ax.axes[0]):
        graph.set(xlabel='Method', ylabel='Marginal coverage', title='SNR: '+str(SNR[i])+' db')
        graph.axhline(1 - alpha, ls='--', color="red")

    ax.savefig("Marginal.png")


    # plot conditional coverage results
    ax = sns.catplot(x="Black box", y="Conditional coverage",
                     hue="Method", col="noise_norm",
                     data=results, kind="box",
                     height=4, aspect=.7)
    # ax = sns.boxplot(y="Conditional coverage", x="Black box", hue="Method", data=results)
    for i, graph in enumerate(ax.axes[0]):
        graph.set(xlabel='Method', ylabel='Conditional coverage', title='SNR: '+str(SNR[i])+' db')
        graph.axhline(1 - alpha, ls='--', color="red")

    ax.savefig("Conditional.png")

    # plot interval size results
    ax = sns.catplot(x="Black box", y="Size",
                     hue="Method", col="noise_norm",
                     data=results, kind="box",
                     height=4, aspect=.7)
    # ax = sns.boxplot(y="Size cover", x="Black box", hue="Method", data=results)
    for i, graph in enumerate(ax.axes[0]):
        graph.set(xlabel='Method', ylabel='Set Size',title='SNR: '+str(SNR[i])+' db')

    ax.savefig("Size.png")

    # plot marginal covarage vs noise
    plt.figure()
    sns.lineplot(data=results, x="noise_norm", y="Coverage", hue="Method", style="Black box", err_style="bars", ci=75)
    plt.grid()
    plt.xlabel("noise_norm")
    plt.ylabel("marginal covarage")
    plt.title("marginal covarage vs noise")

    # plot conditional covarage vs noise
    plt.figure()
    sns.lineplot(data=results, x="noise_norm", y="Conditional coverage", hue="Method", style="Black box",
                 err_style="bars", ci=75)
    plt.grid()
    plt.xlabel("noise_norm")
    plt.ylabel("conditional covarage")
    plt.title("conditional covarage vs noise")

    # plot covarage size vs noise
    plt.figure()
    sns.lineplot(data=results, x="noise_norm", y="Size cover", hue="Method", style="Black box", err_style="bars", ci=75)
    plt.grid()
    plt.xlabel("noise_norm")
    plt.ylabel("prediction set size")
    plt.title("prediction set size vs noise")
    #plt.show()
