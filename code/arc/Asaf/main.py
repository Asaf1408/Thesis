import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import arc
import multiprocessing as mp
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
    black_boxes, methods, X_train, y_train, X_test, y_test, box_name, method_name, noise_std, alpha ,random_state = args

    # get dimension of data
    p = len(X_train[0,:])

    # get size of train and test sets
    n_test = len(y_test)
    n_train = len(y_train)

    # add noise to test set
    X_test = X_test + (noise_std * np.random.randn(n_test,p))

    # get current classifier
    black_box = black_boxes[box_name]

    # Train classification method
    method = methods[method_name](X_train, y_train, black_box, alpha, random_state=random_state, verbose=False)

    # Apply classification method
    S = method.predict(X_test)

    # Evaluate results
    res = evaluate_predictions(S, X_test, y_test)

    res['Method'] = method_name
    res['Black box'] = box_name
    res['Nominal'] = 1 - alpha
    res['n_train'] = n_train
    res['n_test'] = n_test
    res['noise_std'] = noise_std

    return res


def run_experiment(data_model, n_train, n_test, methods, black_boxes, noises, condition_on,
                   alpha=0.1, experiment=0, random_state=2020):

    # Set random seed
    np.random.seed(random_state)

    # Total number of samples
    n = n_train + n_test

    # Generate data with labels
    X = data_model.sample_X(n)
    y = data_model.sample_Y(X)

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=random_state)

    # create parameter list for each run
    parametrs = []
    for box_name in black_boxes:
        for method_name in methods:
            for noise_std in noises:
                parametrs.append((black_boxes, methods, X_train, y_train, X_test, y_test, box_name, method_name, noise_std, alpha, random_state))

    # check for number of cores
    workers = mp.cpu_count()
    print("number of workers: "+str(workers))

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


if __name__ == '__main__':

    plt.close('all')

    import warnings
    warnings.filterwarnings("ignore")

    model_num = 1       # define model for generating data
    alpha = 0.1         # define desired conditional coverage (1-\alpha)
    n_train = 1000      # define number of samples in the training set
    n_test = 5000
    n_experiments = 10  # define Number of independent experiments
    condition_on = [0]  # define features to condition on
    noises = [0,10e-4,10e-3,10e-2,10e-1]
    noises = [0,1e-1,5*(1e-1),1e0,5*(1e0)]

    # Define data model
    if model_num == 1:
        K = 10
        p = 10
        data_model = arc.models.Model_Ex1(K, p)
    else:
        K = 4
        p = 5
        data_model = arc.models.Model_Ex2(K, p)

    # List of calibration methods to be compared
    methods = {
        'None': arc.methods.No_Calibration,
        'SC': arc.methods.SplitConformal,
        'CV+': arc.methods.CVPlus,
       #'JK+': arc.methods.JackknifePlus,
        'HCC': arc.others.SplitConformalHomogeneous,
       #'CQC': arc.others.CQC
        }

    # List of black boxes to be compared
    black_boxes = {
        'Oracle': arc.black_boxes.Oracle(data_model),
        'SVC': arc.black_boxes.SVC(clip_proba_factor=1e-5, random_state=2020),
        #'RFC': arc.black_boxes.RFC(clip_proba_factor=1e-5, n_estimators=1000, max_depth=5, max_features=None,random_state=2020)
    }

    # create dataframe for storing the results
    results = pd.DataFrame()

    # run experiments with all the methods and black boxes
    for experiment in tqdm(range(n_experiments)):
        print("Experiment " + str(experiment) + ":")
        # Random state for this experiment
        random_state = 2020 + experiment

        res = run_experiment(data_model, n_train, n_test, methods, black_boxes, noises, condition_on,
                                       alpha=alpha, experiment=experiment, random_state=random_state)
        results = results.append(res)

    # plot marginal coverage results
    plt.figure()
    ax = sns.catplot(x="Black box", y="Coverage",
                hue="Method", col="noise_std",
                data=results, kind="box",
                height=4, aspect=.7)
    #ax = sns.boxplot(y="Coverage", x="Black box", hue="Method", data=results)
    for graph in ax.axes[0]:
        graph.set(xlabel='Method', ylabel='Marginal coverage')
        graph.axhline(1 - alpha, ls='--', color="red")

    # plot conditional coverage results
    plt.figure()
    ax = sns.catplot(x="Black box", y="Conditional coverage",
                hue="Method", col="noise_std",
                data=results, kind="box",
                height=4, aspect=.7)
    #ax = sns.boxplot(y="Conditional coverage", x="Black box", hue="Method", data=results)
    for graph in ax.axes[0]:
        graph.set(xlabel='Method', ylabel='Conditional coverage')
        graph.axhline(1 - alpha, ls='--', color="red")

    # plot interval size results
    plt.figure()
    ax = sns.catplot(x="Black box", y="Size cover",
                hue="Method", col="noise_std",
                data=results, kind="box",
                height=4, aspect=.7)
    #ax = sns.boxplot(y="Size cover", x="Black box", hue="Method", data=results)
    for graph in ax.axes[0]:
        graph.set(xlabel='Method', ylabel='Conditional Size')

    # plot marginal covarage vs noise
    plt.figure()
    sns.lineplot(data=results, x="noise_std", y="Coverage", hue="Method", style="Black box",err_style="bars", ci=75)
    plt.xscale('log')
    plt.grid()
    plt.xlabel("noise std")
    plt.ylabel("marginal covarage")

    # plot conditional covarage vs noise
    plt.figure()
    sns.lineplot(data=results, x="noise_std", y="Conditional coverage", hue="Method", style="Black box",err_style="bars", ci=75)
    plt.xscale('log')
    plt.grid()
    plt.xlabel("noise std")
    plt.ylabel("conditional covarage")

    # plot covarage size vs noise
    plt.figure()
    sns.lineplot(data=results, x="noise_std", y="Size cover", hue="Method", style="Black box",err_style="bars", ci=75)
    plt.xscale('log')
    plt.grid()
    plt.show()
    plt.xlabel("noise std")
    plt.ylabel("prediction set size")
