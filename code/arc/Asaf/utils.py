import torch
from third_party.smoothing_adversarial.attacks import PGD_L2, DDN
import numpy as np
import arc
import pandas as pd
from torch.nn.functional import softmax
from scipy.stats import rankdata
from Asaf.My_Models import MnistCNN, Cifar10CNN, Cifar100CNN, ResNet
import torch.nn as nn
import time
import os
from numpy.random import default_rng
from scipy.stats.mstats import mquantiles
from scipy.stats import norm
from tqdm import tqdm


# function to calculate accuracy of the model
def calculate_accuracy(model, dataloader, device):
    model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    model_accuracy = total_correct / total_images
    return model_accuracy


def Smooth_Adv(model, x, y, noises, N_steps=20, max_norm=0.125, device='cpu', method='PGD'):
    # create attack model
    if method == 'PGD':
        attacker = PGD_L2(steps=N_steps, device=device, max_norm=max_norm)
    elif method == "DDN":
        attacker = DDN(steps=N_steps, device=device, max_norm=max_norm)

    # create container for the adversarial examples
    x_adv = torch.zeros_like(x)

    # get number of data points
    n = x.size()[0]

    # number of permutations to estimate mean
    num_of_noise_vecs = noises.size()[0] // n

    # calculate maximum batch size according to gpu capacity
    batch_size = 1024 // num_of_noise_vecs

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # start generating examples for each batch
    print("Generating Adverserial Examples:")

    for j in tqdm(range(num_of_batches)):
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]

        # duplicate batch according to the number of added noises and send to device
        # the first num_of_noise_vecs samples will be duplicates of x[0] and etc.
        tmp = torch.zeros((len(labels) * num_of_noise_vecs, *inputs.shape[1:]))
        x_tmp = inputs.repeat((1, num_of_noise_vecs, 1, 1)).view(tmp.shape).to(device)

        # send labels to device
        y_tmp = labels.to(device).long()

        # generate random Gaussian noise for the duplicated batch
        noise = noises[(j * (batch_size * num_of_noise_vecs)):((j + 1) * (batch_size * num_of_noise_vecs))].to(device)
        # noise = torch.randn_like(x_tmp, device=device) * sigma_adv

        # generate adversarial examples for the batch
        x_adv_batch = attacker.attack(model, x_tmp, y_tmp,
                                      noise=noise, num_noise_vectors=num_of_noise_vecs,
                                      no_grad=False,
                                      )

        # take only the one example for each point
        x_adv_batch = x_adv_batch[::num_of_noise_vecs]

        # put in the container
        x_adv[(j * batch_size):((j + 1) * batch_size)] = x_adv_batch

    # move adversarial examples back to cpu
    x_adv = x_adv.to(torch.device('cpu'))

    # return adversarial examples
    return x_adv


def evaluate_predictions(S, X, y, conditional=True):
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


# calculate accuracy of the smoothed classifier
def calculate_accuracy_smooth(model, x, y, noises, num_classes, k=1, device='cpu'):
    # get size of the test et
    n = x.size()[0]

    # number of permutations to estimate mean
    n_smooth = noises.size()[0] // n

    # create container for the outputs
    smoothed_predictions = torch.zeros((n, num_classes))

    # calculate maximum batch size according to gpu capacity
    batch_size = 1024 // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # get predictions over all batches
    for j in range(num_of_batches):
        # get inputs and labels of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]
        labels = y[(j * batch_size):((j + 1) * batch_size)]

        # duplicate batch according to the number of added noises and send to device
        # the first n_smooth samples will be duplicates of x[0] and etc.
        tmp = torch.zeros((len(labels) * n_smooth, *inputs.shape[1:]))
        x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

        # generate random Gaussian noise for the duplicated batch
        noise = noises[(j * (batch_size * n_smooth)):((j + 1) * (batch_size * n_smooth))].to(device)

        # add noise to points
        noisy_points = x_tmp + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1)

        # get smoothed prediction for each point
        for m in range(len(labels)):
            smoothed_predictions[(j * batch_size) + m, :] = torch.mean(
                noisy_outputs[(m * n_smooth):((m + 1) * n_smooth)], dim=0)

    # transform results to numpy array
    smoothed_predictions = smoothed_predictions.numpy()

    # get label ranks to calculate top k accuracy
    label_ranks = np.array([rankdata(-smoothed_predictions[i, :], method='ordinal')[y[i]] - 1 for i in range(n)])

    # calculate accuracy
    top_k_accuracy = np.sum(label_ranks <= (k - 1)) / float(n)

    # return accuracy
    return top_k_accuracy


def train_loop(train_loader, test_loader, dataset='CIFAR10', num_epochs=50, robust_epoches=0, learning_rate=0.001,
               sigma=0, device='cpu'):
    # build our model and send it to the device
    if dataset == "MNIST":
        model = MnistCNN().to(device)
    elif dataset == "CIFAR10":
        model = Cifar10CNN().to(device)
    elif dataset == "CIFAR100":
        model = ResNet(depth=110).to(device)
        #model = Cifar100CNN().to(device)

    num_of_classes = 10
    noises = []
    # set loss criterion
    criterion = nn.CrossEntropyLoss()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    robust_epoches = np.min([robust_epoches, num_epochs])

    # training loop
    for epoch in range(1, num_epochs + 1):
        model.train()  # put in training mode
        running_loss = 0.0
        epoch_time = time.time()
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # add Gaussian noise augmentation
            inputs = inputs + (torch.randn_like(inputs) * sigma)

            # send labels to device
            labels = labels.to(device)

            if epoch > (num_epochs - robust_epoches):
                # get number of points in the batch
                points_in_batch = inputs.size()[0]

                # create container for mean outputs
                outputs = torch.zeros(points_in_batch, num_of_classes).to(device)

                # forward
                # estimate mean over all noise added points
                for j in range(points_in_batch):
                    # add noise to data point
                    noisy_points = inputs[j, :, :, :] + noises

                    # get classifier predictions
                    noisy_points = noisy_points.to(device)
                    noisy_outputs = model(noisy_points)

                    # calculate mean over all outputs
                    outputs[j, :] = torch.mean(noisy_outputs)

            else:
                # send inputs to device
                inputs = inputs.to(device)

                # forward
                outputs = model(inputs)

            # backward + optimize
            loss = criterion(outputs, labels)  # calculate the loss

            # always the same 3 steps
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters
            # print statistics
            running_loss += loss.data.item()

        # Normalizing the loss by the total number of train batches
        running_loss /= len(train_loader)

        # Calculate training/test set accuracy of the existing model
        train_accuracy = calculate_accuracy(model, train_loader, device)
        test_accuracy = calculate_accuracy(model, test_loader, device)

        # print epoch information
        log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Test accuracy: {:.3f}% | ".format(epoch
                                                                                                         , running_loss,
                                                                                                         100 * train_accuracy,
                                                                                                         100 * test_accuracy)
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)

    print('==> Finished Training ...')

    # save model
    print('==> Saving model ...')
    state = {'net': model.state_dict(), 'epoch': num_epochs}
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if dataset == "MNIST":
        torch.save(state, './checkpoints/MnistCNN_sigma_' + str(sigma) + '.pth')
    elif dataset == "CIFAR10":
        torch.save(state, './checkpoints/Cifar10CNN_sigma_' + str(sigma) + '.pth')
    elif dataset == "CIFAR100":
        torch.save(state, './checkpoints/Cifar100CNN_sigma_' + str(sigma) + '.pth')

    return model


def smooth_calibration(model, x_calib, y_calib, noises, alpha, num_of_classes, scores_list, correction, base=False, device='cpu'):
    # size of the calibration set
    n_calib = x_calib.size()[0]

    # number of permutations to estimate mean
    n_smooth = noises.size()[0] // n_calib

    # create container for the scores
    if base:
        scores_simple = np.zeros((len(scores_list), n_calib))
    else:
        smoothed_scores = np.zeros((len(scores_list), n_calib))
        scores_smoothed = np.zeros((len(scores_list), n_calib))

    # create container for the calibration thresholds
    thresholds = np.zeros((len(scores_list), 3))

    # calculate maximum batch size according to gpu capacity
    batch_size = 1024 // n_smooth

    # calculate number of batches
    if n_calib % batch_size != 0:
        num_of_batches = (n_calib // batch_size) + 1
    else:
        num_of_batches = (n_calib // batch_size)

    # create container for smoothed and base classifier outputs
    if base:
        simple_outputs = np.zeros((n_calib, num_of_classes))
    else:
        smooth_outputs = np.zeros((n_calib, num_of_classes))

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n_calib, low=0.0, high=1.0)

    # pass all points to model in batches and calculate scores
    for j in range(num_of_batches):
        # get inputs and labels of batch
        inputs = x_calib[(j * batch_size):((j + 1) * batch_size)]
        labels = y_calib[(j * batch_size):((j + 1) * batch_size)]

        if base:
            noise = noises[(j * batch_size):((j + 1) * batch_size)].to(device)
            noisy_points = inputs.to(device) + noise
        else:
            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((len(labels) * n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

            # generate random Gaussian noise for the duplicated batch
            noise = noises[(j * (batch_size * n_smooth)):((j + 1) * (batch_size * n_smooth))].to(device)

            # add noise to points
            noisy_points = x_tmp + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        # get smoothed score for each point
        if base:
            simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
        else:
            for k in range(len(labels)):

                # get all the noisy outputs of a specific point
                point_outputs = noisy_outputs[(k * n_smooth):((k + 1) * n_smooth)]

                # get smoothed classifier output of this point
                smooth_outputs[(j * batch_size) + k, :] = np.mean(point_outputs, axis=0)

                # get smoothed score of this point

                # generate random variable for inverse quantile score
                u = np.ones(n_smooth) * uniform_variables[(j * batch_size) + k]

                # run over all scores functions and compute smoothed scores
                for p, score_func in enumerate(scores_list):
                    # get smoothed score
                    tmp_scores = score_func(point_outputs, labels[k], u, all_combinations=True)
                    smoothed_scores[p, (j * batch_size) + k] = np.mean(tmp_scores)

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        if base:
            scores_simple[p, :] = score_func(simple_outputs, y_calib, uniform_variables, all_combinations=False)
        else:
            scores_smoothed[p, :] = score_func(smooth_outputs, y_calib, uniform_variables, all_combinations=False)

    # Compute thresholds
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n_calib))

    bounds = np.zeros((len(scores_list), 2))
    for p in range(len(scores_list)):
        if base:
            thresholds[p, 0] = mquantiles(scores_simple[p, :], prob=level_adjusted)
        else:
            thresholds[p, 1] = mquantiles(scores_smoothed[p, :], prob=level_adjusted)
            thresholds[p, 2] = mquantiles(smoothed_scores[p, :], prob=level_adjusted)

            # calculate lower and upper bounds of correction of smoothed score
            upper_thresh = norm.cdf(norm.ppf(thresholds[p, 2], loc=0, scale=1)+correction, loc=0, scale=1)
            lower_thresh = norm.cdf(norm.ppf(thresholds[p, 2], loc=0, scale=1)-correction, loc=0, scale=1)

            bounds[p, 0] = np.size(smoothed_scores[p, :][smoothed_scores[p, :] <= lower_thresh])/np.size(smoothed_scores[p, :])
            bounds[p, 1] = np.size(smoothed_scores[p, :][smoothed_scores[p, :] <= upper_thresh]) / np.size(smoothed_scores[p, :])

    return thresholds, bounds


def predict_sets(model, x, noises, num_of_classes, scores_list, thresholds, correction, base=False, device='cpu'):
    # get number of points
    n = x.size()[0]

    # number of permutations to estimate mean
    n_smooth = noises.size()[0] // n

    # create container for the scores
    if base:
        scores_simple = np.zeros((len(scores_list), n, num_of_classes))
    else:
        smoothed_scores = np.zeros((len(scores_list), n, num_of_classes))
        scores_smoothed = np.zeros((len(scores_list), n, num_of_classes))

    # calculate maximum batch size according to gpu capacity
    batch_size = 1024 // n_smooth

    # calculate number of batches
    if n % batch_size != 0:
        num_of_batches = (n // batch_size) + 1
    else:
        num_of_batches = (n // batch_size)

    # initiate random uniform variables for inverse quantile score
    rng = default_rng()
    uniform_variables = rng.uniform(size=n, low=0.0, high=1.0)

    # create container for smoothed and base classifier outputs
    if base:
        simple_outputs = np.zeros((n, num_of_classes))
    else:
        smooth_outputs = np.zeros((n, num_of_classes))

    for j in range(num_of_batches):
        # get inputs of batch
        inputs = x[(j * batch_size):((j + 1) * batch_size)]

        if base:
            noise = noises[(j * batch_size):((j + 1) * batch_size)].to(device)
            noisy_points = inputs.to(device) + noise
        else:
            # duplicate batch according to the number of added noises and send to device
            # the first n_smooth samples will be duplicates of x[0] and etc.
            tmp = torch.zeros((inputs.size()[0] * n_smooth, *inputs.shape[1:]))
            x_tmp = inputs.repeat((1, n_smooth, 1, 1)).view(tmp.shape).to(device)

            # generate random Gaussian noise for the duplicated batch
            noise = noises[(j * (batch_size * n_smooth)):((j + 1) * (batch_size * n_smooth))].to(device)

            # add noise to points
            noisy_points = x_tmp + noise

        # get classifier predictions on noisy points
        model.eval()  # put in evaluation mode
        with torch.no_grad():
            noisy_outputs = model(noisy_points).to(torch.device('cpu'))

        # transform the output into probabilities vector
        noisy_outputs = softmax(noisy_outputs, dim=1).numpy()

        if base:
            simple_outputs[(j * batch_size):((j + 1) * batch_size), :] = noisy_outputs
        else:
            # get smoothed score for each point
            for k in range(inputs.size()[0]):

                # get all the noisy outputs of a specific point
                point_outputs = noisy_outputs[(k * n_smooth):((k + 1) * n_smooth)]

                # get smoothed classifier output of this point
                smooth_outputs[(j * batch_size) + k, :] = np.mean(point_outputs, axis=0)

                # generate random variable for inverse quantile score
                u = np.ones(n_smooth) * uniform_variables[(j * batch_size) + k]

                # run over all scores functions and compute smoothed scores with all lables
                for p, score_func in enumerate(scores_list):
                    smoothed_scores[p, ((j * batch_size) + k), :] = np.mean(
                        score_func(point_outputs, np.arange(num_of_classes), u, all_combinations=True), axis=0)

    # run over all scores functions and compute scores of smoothed and base classifier
    for p, score_func in enumerate(scores_list):
        if base:
            scores_simple[p, :, :] = score_func(simple_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)
        else:
            scores_smoothed[p, :, :] = score_func(smooth_outputs, np.arange(num_of_classes), uniform_variables, all_combinations=True)

    # Generate prediction sets using the thresholds from the calibration
    predicted_sets = []
    for p in range(len(scores_list)):
        if base:
            S_hat_simple = [np.where(norm.ppf(scores_simple[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 0], loc=0, scale=1))[0] for i in range(n)]
            predicted_sets.append(S_hat_simple)
        else:
            S_hat_smoothed = [np.where(norm.ppf(scores_smoothed[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 1], loc=0, scale=1))[0] for i in range(n)]
            smoothed_S_hat = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) <= norm.ppf(thresholds[p, 2], loc=0, scale=1))[0] for i in range(n)]
            smoothed_S_hat_corrected = [np.where(norm.ppf(smoothed_scores[p, i, :], loc=0, scale=1) - correction <= norm.ppf(thresholds[p, 2], loc=0, scale=1))[0] for i in range(n)]

            tmp_list = [S_hat_smoothed, smoothed_S_hat, smoothed_S_hat_corrected]
            predicted_sets.append(tmp_list)

    # return predictions sets
    return predicted_sets
