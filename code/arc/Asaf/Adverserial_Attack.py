from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import torch
import torchvision
import os
import pickle

from numpy.random import default_rng

from torch.utils.data.dataset import random_split

# from art.attacks.evasion import FastGradientMethod
# from art.attacks.evasion import AutoProjectedGradientDescent
# from art.attacks.evasion import ProjectedGradientDescent
#from art.attacks.evasion import CarliniL2Method
#from art.attacks.evasion import DeepFool

# from art.estimators.classification import PyTorchClassifier
# from art.utils import load_mnist
# from art.utils import load_cifar10

from third_party.smoothing_adversarial.architectures import get_architecture
from third_party.smoothing_adversarial.attacks import PGD_L2, DDN

import pandas as pd
import Asaf.Score_Functions as scores
import Asaf.methods_new as methods
from Asaf.My_Models import MnistCNN, Cifar10CNN, Cifar100CNN, ResNet
from Asaf.utils import calculate_accuracy, Smooth_Adv, evaluate_predictions, calculate_accuracy_smooth, train_loop, smooth_calibration, predict_sets
from sklearn.model_selection import train_test_split

alpha = 0.1                         # desired nominal marginal coverage
epsilon = 0.125                     # L2 bound on the adversarial noise
n_experiments = 50                  # number of experiments to estimate coverage
n_test = 10000                       # number of test points (if larger then available it takes the entire set)
ratio = 2                           # ratio between adversarial noise bound to smoothed noise
train = False                        # whether to train a model or not
sigma_smooth = ratio * epsilon      # sigma used fro smoothing
sigma_model = sigma_smooth                 # sigma used for training the model
n_smooth = 132                        # number of samples used for smoothing
My_model = True                     # use my model or salman/cohen models
N_steps = 20                        # number of gradiant steps for PGD attack
dataset = "CIFAR100"                 # dataset to be used 'MNIST', 'CIFAR100', 'CIFAR10', 'ImageNet'
calibration_scores = ['HCC', 'SC', 'SC_Reg']  # score function to check 'HCC', 'SC', 'SC_Reg'

# calculate correction based on the Lipschitz constant
if sigma_smooth == 0:
    correction = 10000
else:
    correction = float(epsilon) / float(sigma_smooth)

# hyper-parameters for training if we want to train a model
if train:
    num_epochs = 20         # number of epochs for training
    learning_rate = 1e-4   # learning rate of the optimizer
    batch_size = 128        # batch size of the data loaders
    robust_epochs = 0       # number of robust epochs for training

# set random seed according to time
seed = int(time.time())
torch.manual_seed(seed)

# load datasets
if dataset == "MNIST":
    # Load train set
    train_dataset = torchvision.datasets.MNIST(root='./datasets/',
                                               train=True,
                                               transform=torchvision.transforms.ToTensor(),
                                               download=True)
    # load test set
    test_dataset = torchvision.datasets.MNIST(root='./datasets',
                                              train=False,
                                              transform=torchvision.transforms.ToTensor())

elif dataset == "CIFAR10":
    # Load train set
    train_dataset = torchvision.datasets.CIFAR10(root='./datasets/',
                                                 train=True,
                                                 transform=torchvision.transforms.ToTensor(),
                                                 download=True)
    # load test set
    test_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                train=False,
                                                transform=torchvision.transforms.ToTensor())

elif dataset == "CIFAR100":
    # Load train set
    train_dataset = torchvision.datasets.CIFAR100(root='./datasets/',
                                                 train=True,
                                                 transform=torchvision.transforms.ToTensor(),
                                                 download=True)
    # load test set
    test_dataset = torchvision.datasets.CIFAR100(root='./datasets',
                                                train=False,
                                                transform=torchvision.transforms.ToTensor())

elif dataset == "ImageNet":
    # Load train set
    train_dataset = torchvision.datasets.ImageNet(root='./datasets/',
                                                 train=True,
                                                 transform=torchvision.transforms.ToTensor(),
                                                 download=True)
    # load test set
    test_dataset = torchvision.datasets.ImageNet(root='./datasets',
                                                train=False,
                                                transform=torchvision.transforms.ToTensor())

# cut the size of the test set if necessary
if n_test < len(test_dataset):
    test_dataset = torch.utils.data.random_split(test_dataset, [n_test, len(test_dataset) - n_test])[0]

# save the sizes of each one of the sets
n_train = len(train_dataset)
n_test = len(test_dataset)

# Create Data loader for train set if training is needed
if train:
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)

# Create Data loader for test set
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=n_test,
                                          shuffle=False)

# convert test set into tensor
examples = enumerate(test_loader)
batch_idx, (x_test, y_test) = next(examples)

# get dimension of data
rows = x_test.size()[2]
cols = x_test.size()[3]
channels = x_test.size()[1]
num_of_classes = len(train_dataset.classes)
min_pixel_value = 0.0
max_pixel_value = 1.0

# automatically choose device use gpu 0 if it is available o.w. use the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print the chosen device
print("device: ", device)

# training a model on the desired dataset
if train:
    model = train_loop(train_loader, test_loader, dataset, num_epochs, robust_epochs, learning_rate, sigma_model, device)

# loading a pre-trained model
else:
    # load my models
    if My_model:
        if dataset == "MNIST":
            model = MnistCNN()
            state = torch.load('./checkpoints/MnistCNN_sigma_'+str(sigma_model)+'.pth')
            model.load_state_dict(state['net'])
        elif dataset == "CIFAR10":
            state = torch.load('./checkpoints/Cifar10CNN_sigma_'+str(sigma_model)+'.pth')
            model = Cifar10CNN()
            model.load_state_dict(state['net'])
        elif dataset == "CIFAR100":
            state = torch.load('./checkpoints/ResNet110_sigma_'+str(sigma_model)+'.pth.tar')
            model = ResNet(depth=110)
            model.load_state_dict(state['state_dict'])
        elif dataset == "ImageNet":
            state = torch.load('./checkpoints/ImageNetCNN_sigma_'+str(sigma_model)+'.pth')
            model = Cifar100CNN()

    # load cohen and salman models
    else:
        #checkpoint = torch.load(
        #   './pretrained_models/Salman/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_'+str(sigma_model)+'/checkpoint.pth.tar')
        if dataset == "CIFAR10":
            checkpoint = torch.load('./pretrained_models/Cohen/cifar10/resnet110/noise_'+str(sigma_model)+'/checkpoint.pth.tar')
            model = get_architecture(checkpoint["arch"], "cifar10")
        elif dataset == "ImageNet":
            checkpoint = torch.load('./pretrained_models/Cohen/imagenet/resnet50/noise_'+str(sigma_model)+'/checkpoint.pth.tar')
            model = get_architecture(checkpoint["arch"], "imagenet")
        model.load_state_dict(checkpoint['state_dict'])

# send model to device
model.to(device)

# put model in evaluation mode
model.eval()

# create indices for the test points
indices = torch.arange(n_test)

directory = "./Adversarial_Examples/"+str(dataset)+"/sigma_model_"+str(sigma_model)+"/n_smooth_"+str(n_smooth)

if not os.path.exists(directory) or not n_test == 10000:

    # create noises for each data point
    noises_test = torch.randn((n_test * n_smooth, channels, rows, cols)) * sigma_smooth

    # create noises for the base classifier
    noises_test_base = torch.randn((n_test, channels, rows, cols)) * sigma_model

    # Generate adversarial test examples
    x_test_adv = Smooth_Adv(model, x_test, y_test, noises_test, N_steps, epsilon, device)

    # Generate adversarial test examples for the base classifier
    x_test_adv_base = Smooth_Adv(model, x_test, y_test, noises_test_base, N_steps, epsilon, device)

    if n_test == 10000:
        os.makedirs(directory)
        with open(directory+"/data.pickle", 'wb') as f:
            pickle.dump([x_test_adv, x_test_adv_base, noises_test, noises_test_base], f)

else:
    with open(directory+"/data.pickle", 'rb') as f:
        x_test_adv, x_test_adv_base, noises_test, noises_test_base = pickle.load(f)

# n_to_check = np.array([1, 2, 4])
# acuuracy_true = np.zeros_like(n_to_check)
# acuuracy_adv_PGD = np.zeros_like(n_to_check)
# acuuracy_adv_DDN = np.zeros_like(n_to_check)
#
# for j, n_smooth in enumerate(n_to_check):
#     # generate random Gaussian noises for smoothing, the same vectors will bu used for attacking
#     noises = torch.randn((n_test*n_smooth, channels, rows, cols)) * sigma_smooth
#
#     # compute accuracy on the test set
#     accuracy = calculate_accuracy_smooth(model, x_test, y_test, noises, num_classes=num_of_classes, k=1, device=device)
#     print("Accuracy on test set: {}%".format(accuracy * 100))
#     acuuracy_true[j] = accuracy * 100
#     # # Create the ART classifier wrapper based on the model
#     # classifier = PyTorchClassifier(
#     #    model=model,
#     #    clip_values=(min_pixel_value, max_pixel_value),
#     #    loss=criterion,
#     #    optimizer=optimizer,
#     #    input_shape=(channels, rows, cols),
#     #    nb_classes=num_of_classes,
#     # )
#
#     # Generate adversarial test examples
#     # attack = FastGradientMethod(estimator=classifier, eps=epsilon, norm=2)
#     # attack = AutoProjectedGradientDescent(estimator=classifier, eps=epsilon, norm=2)
#     # attack = ProjectedGradientDescent(estimator=classifier, eps=epsilon, norm=2, batch_size=batch_size)
#     # attack = CarliniL2Method(classifier=classifier, batch_size=batch_size)
#     # attack = DeepFool(classifier=classifier, batch_size=batch_size)
#     # x_test_adv = torch.from_numpy(attack.generate(x=x_test.numpy()))
#
#     # generate adversarial example with PGD for smoothed classifiers
#     x_test_adv = Smooth_Adv(model, x_test, y_test, noises, N_steps, epsilon, device, 'PGD')
#
#     # compute accuracy on the adversarial test set
#     accuracy = calculate_accuracy_smooth(model, x_test_adv, y_test, noises, num_classes=num_of_classes, k=1, device=device)
#     print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
#     acuuracy_adv_PGD[j] = accuracy * 100
#
#     # generate adversarial example with DDN for smoothed classifiers
#     x_test_adv = Smooth_Adv(model, x_test, y_test, noises, N_steps, epsilon, device, 'DDN')
#
#     # compute accuracy on the adversarial test set
#     accuracy = calculate_accuracy_smooth(model, x_test_adv, y_test, noises, num_classes=num_of_classes, k=1, device=device)
#     print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
#     acuuracy_adv_DDN[j] = accuracy * 100
#
# plt.figure()
# plt.plot(n_to_check, acuuracy_true, color='red', label="test set")
# plt.plot(n_to_check, acuuracy_adv_PGD, color='blue', label="adversarial PGD-20 test set")
# plt.plot(n_to_check, acuuracy_adv_DDN, color='green', label="adversarial PGD-50 test set")
#
# plt.xlabel("number of noises")
# plt.ylabel("accuracy")
# plt.grid()
# plt.legend()
# plt.savefig("Smoothing_Comparison.png")

# translate desired scores to their functions and put in a list
scores_list = []
for score in calibration_scores:
    if score == 'HCC':
        scores_list.append(scores.class_probability_score)
    if score == 'SC':
        scores_list.append(scores.generalized_inverse_quantile_score)
    if score == 'SC_Reg':
        scores_list.append(scores.rank_regularized_score)

acc = calculate_accuracy_smooth(model, x_test, y_test, noises_test_base, num_of_classes, k=1, device=device)
print("True Model accuracy :" + str(acc*100) + "%")

acc = calculate_accuracy_smooth(model, x_test_adv_base, y_test, noises_test_base, num_of_classes, k=1, device=device)
print("True Model accuracy on adversarial examples :" + str(acc*100) + "%")
#exit(1)

# create dataframe for storing results
results = pd.DataFrame()
quantiles = np.zeros((len(scores_list), 2, n_experiments))
for experiment in tqdm(range(n_experiments)):

    # Split test data into calibration and test
    x_calib, x_test_new, y_calib, y_test_new, idx1, idx2 = train_test_split(x_test, y_test, indices, test_size=0.5)

    # save sizes of calibration and test sets
    n_calib = x_calib.size()[0]
    n_test_new = x_test_new.size()[0]

    # get the relevant noises for the calibration and test sets
    noises_calib = torch.zeros((n_calib*n_smooth, channels, rows, cols))
    noises_calib_base = noises_test_base[idx1]
    noises_test_new = torch.zeros((n_test_new*n_smooth, channels, rows, cols))
    noises_test_new_base = noises_test_base[idx2]

    for j, m in enumerate(idx1):
        noises_calib[(j*n_smooth):((j+1)*n_smooth), :, :, :] = noises_test[(m*n_smooth):((m+1)*n_smooth), :, :, :]

    for j, m in enumerate(idx2):
        noises_test_new[(j*n_smooth):((j+1)*n_smooth), :, :, :] = noises_test[(m*n_smooth):((m+1)*n_smooth), :, :, :]

    # get the relevant adversarial examples for the new test set
    x_test_adv_new = x_test_adv[idx2]
    x_test_adv_new_base = x_test_adv_base[idx2]

    # calibrate the model with the desired scores and get the thresholds
    thresholds, bounds = smooth_calibration(model, x_calib, y_calib, noises_calib, alpha, num_of_classes, scores_list, correction, base=False, device=device)

    # calibrate base model with the desired scores and get the thresholds
    thresholds_base, _ = smooth_calibration(model, x_calib, y_calib, noises_calib_base, alpha, num_of_classes, scores_list, correction, base=True, device=device)

    thresholds = thresholds + thresholds_base

    # put bounds in array of bounds
    for p in range(len(scores_list)):
        quantiles[p, 0, experiment] = bounds[p, 0]
        quantiles[p, 1, experiment] = bounds[p, 1]

    # generate prediction sets on the clean test set
    predicted_clean_sets = predict_sets(model, x_test_new, noises_test_new, num_of_classes, scores_list, thresholds, correction, base=False, device=device)

    # generate prediction sets on the clean test set
    predicted_clean_sets_base = predict_sets(model, x_test_new, noises_test_new_base, num_of_classes, scores_list, thresholds, correction, base=True, device=device)

    # generate prediction sets on the adversarial test set
    predicted_adv_sets = predict_sets(model, x_test_adv_new, noises_test_new, num_of_classes, scores_list, thresholds, correction, base=False, device=device)

    # generate prediction sets on the adversarial test set
    predicted_adv_sets_base = predict_sets(model, x_test_adv_new_base, noises_test_new_base, num_of_classes, scores_list, thresholds, correction, base=True, device=device)

    for p in range(len(scores_list)):
        predicted_clean_sets[p].insert(0, predicted_clean_sets_base[p])
        predicted_adv_sets[p].insert(0, predicted_adv_sets_base[p])
        score_name = calibration_scores[p]
        methods_list = [score_name+'_simple', score_name+'_smoothed_classifier', score_name+'_smoothed_score', score_name+'_smoothed_score_correction']
        for r, method in enumerate(methods_list):
            res = evaluate_predictions(predicted_clean_sets[p][r], x_test_new.numpy(), y_test_new.numpy(), conditional=False)
            res['Method'] = methods_list[r]
            res['noise_L2_norm'] = 0
            res['Black box'] = 'CNN sigma = '+str(sigma_model)
            # Add results to the list
            results = results.append(res)

    for p in range(len(scores_list)):
        score_name = calibration_scores[p]
        methods_list = [score_name + '_simple', score_name + '_smoothed_classifier', score_name + '_smoothed_score',
                        score_name + '_smoothed_score_correction']
        for r, method in enumerate(methods_list):
            res = evaluate_predictions(predicted_adv_sets[p][r], x_test_new.numpy(), y_test_new.numpy(),
                                        conditional=False)
            res['Method'] = methods_list[r]
            res['noise_L2_norm'] = epsilon
            res['Black box'] = 'CNN sigma = '+str(sigma_model)
            # Add results to the list
            results = results.append(res)

        # res['Method'] = method_name
        # res['Nominal'] = 1 - alpha
        # res['n_test'] = n_test
        # res['n_calib'] = n_calib
        # res['n_train'] = n_train
        # res['noise_norm'] = 0
        # res['Black box'] = 'CNN'


directory = "./Results/"+str(dataset)+"/sigma_model_"+str(sigma_model)+"/sigma_smooth_"+str(sigma_smooth)+"/n_smooth_"+str(n_smooth)

for score in calibration_scores:
    if score == 'SC_Reg':
        directory = directory + "/Regularization"
        break

if not os.path.exists(directory):
    os.makedirs(directory)

# save results
results.to_csv(directory+"/results.csv")

# plot results
# plot marginal coverage results
colors_list = sns.color_palette("husl", len(scores_list)*4)

ax = sns.catplot(x="Black box", y="Coverage",
                 hue="Method", palette=colors_list, col="noise_L2_norm",
                 data=results, kind="box",
                 height=4, aspect=.7)
# ax = sns.boxplot(y="Coverage", x="Black box", hue="Method", data=results)
# upper_quantiles_mean = np.mean(upper_quantiles)
# upper_quantiles_std = np.std(upper_quantiles)
lower_quantiles_mean = np.zeros(len(scores_list))
upper_quantiles_mean = np.zeros(len(scores_list))
lower_quantiles_std = np.zeros(len(scores_list))
upper_quantiles_std = np.zeros(len(scores_list))

for p in range(len(scores_list)):
    lower_quantiles_mean[p] = np.mean(quantiles[p, 0, :])
    upper_quantiles_mean[p] = np.mean(quantiles[p, 1, :])
    lower_quantiles_std[p] = np.std(quantiles[p, 0, :])
    upper_quantiles_std[p] = np.std(quantiles[p, 1, :])

colors = ['green', 'blue']
for i, graph in enumerate(ax.axes[0]):
    graph.set(xlabel='Classifier', ylabel='Marginal coverage')
    graph.axhline(1 - alpha, ls='--', color="red")
    #graph.axhline(upper_quantiles_mean, ls='--', color="green")
    #graph.axhline(lower_quantiles_mean, ls='--', color="green")
    for p in range(len(scores_list)):
        graph.axhline(upper_quantiles_mean[p], ls='--', color=colors_list[p*4+2])
        graph.axhline(lower_quantiles_mean[p], ls='--', color=colors_list[p*4+2])
        #graph.axhspan(upper_quantiles_mean[p]-upper_quantiles_std[p], upper_quantiles_mean[p]+upper_quantiles_std[p], alpha=0.1, color=colors[p])
        #graph.axhspan(lower_quantiles_mean[p]-lower_quantiles_std[p], lower_quantiles_mean[p]+lower_quantiles_std[p], alpha=0.1, color=colors[p])

ax.savefig(directory+"/Marginal.png")

# plot conditional coverage results
ax = sns.catplot(x="Black box", y="Conditional coverage",
                 hue="Method", col="noise_L2_norm",
                 data=results, kind="box",
                 height=4, aspect=.7)
# ax = sns.boxplot(y="Conditional coverage", x="Black box", hue="Method", data=results)
for i, graph in enumerate(ax.axes[0]):
    graph.set(xlabel='Classifier', ylabel='Conditional coverage')
    graph.axhline(1 - alpha, ls='--', color="red")

ax.savefig(directory+"/Conditional.png")

# plot interval size results
ax = sns.catplot(x="Black box", y="Size",
                 hue="Method", col="noise_L2_norm",
                 data=results, kind="box",
                 height=4, aspect=.7)
# ax = sns.boxplot(y="Size cover", x="Black box", hue="Method", data=results)
for i, graph in enumerate(ax.axes[0]):
    graph.set(xlabel='Classifier', ylabel='Set Size')

ax.savefig(directory+"/Size.png")
