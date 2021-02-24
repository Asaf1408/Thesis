"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""

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
from numpy.random import default_rng

from torch.utils.data.dataset import random_split

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import AutoProjectedGradientDescent
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import CarliniL2Method

from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from art.utils import load_cifar10

from third_party.smoothing_adversarial.architectures import get_architecture


import pandas as pd
import Asaf.Score_Functions as scores
import Asaf.methods_new as methods
from sklearn.model_selection import train_test_split


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


# Define the neural network model,
# return logits instead of activation in forward method
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


class CifarCNN(nn.Module):
    """CNN for the CIFAR-10 Datset"""
    def __init__(self):
        """CNN Builder."""
        super(CifarCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """Perform forward."""
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x


alpha = 0.1  # desired nominal marginal coverage
epsilon = 0.5  # L2 bound on the adversarial noise
n_experiments = 10  # number of experiments to estimate coverage
n_test = 2000  # number of test points (if larger then available it takes the entire set)
ratio = 2  # ratio between adversarial noise bound to smoothed noise
train = False

# hyper-parameters:
num_epochs = 10
learning_rate = 0.001
batch_size = 128
dataset = "CIFAR10"

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

# cut the size of the test set if necessary
if n_test < len(test_dataset):
    test_dataset = torch.utils.data.random_split(test_dataset, [n_test, len(test_dataset) - n_test])[0]

# save the sizes of each one of the sets
n_train = len(train_dataset)
n_test = len(test_dataset)

# Create Data loader for train set
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

# set loss criterion
criterion = nn.CrossEntropyLoss()

# build our model and send it to the device
if dataset == "MNIST":
    model = MnistCNN().to(device)
elif dataset == "CIFAR10":
    model = CifarCNN().to(device)

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# generate random vectors from the Gaussian distribution
rng = default_rng()
mean = 0
sigma = ratio * epsilon
n_permutations = 50
robust_epoches = 0
robust_epoches = np.min([robust_epoches, num_epochs])
noises = torch.from_numpy(rng.normal(mean, sigma, (n_permutations, channels, rows, cols)).astype(np.float32))
if (train):
    # training loop
    for epoch in range(1, num_epochs + 1):
        model.train()  # put in training mode
        running_loss = 0.0
        epoch_time = time.time()
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

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
        torch.save(state, './checkpoints/MnistCNN.pth')
    elif dataset == "CIFAR10":
        torch.save(state, './checkpoints/Cifar10CNN.pth')

else:
    if dataset == "MNIST":
        state = torch.load('./checkpoints/MnistCNN.pth', map_location=device)
    elif dataset == "CIFAR10":
        state = torch.load('./checkpoints/Cifar10CNN.pth', map_location=device)
    model.load_state_dict(state['net'])

# load the base classifier
checkpoint = torch.load('./pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_0.12/checkpoint.pth.tar')
model = get_architecture(checkpoint["arch"], "cifar10")
model.load_state_dict(checkpoint['state_dict'])

# initiate random generator
# rng = default_rng()

# generate random vectors from the Gaussian distribution
# noises = rng.normal(0, ratio*epsilon, (n_train, channels, rows, cols)).astype(np.float32)

# add noise to train data
# x_train = x_train + noises


# Create the ART classifier wrapper based on the model
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(channels, rows, cols),
    nb_classes=num_of_classes,
)

# Generate adversarial test examples
# attack = FastGradientMethod(estimator=classifier, eps=epsilon, norm=2)
# attack = AutoProjectedGradientDescent(estimator=classifier, eps=epsilon, norm=2)
attack = ProjectedGradientDescent(estimator=classifier, eps=epsilon, norm=2)
# attack = CarliniL2Method(classifier=classifier)
x_test_adv = torch.from_numpy(attack.generate(x=x_test.numpy()))

# Evaluate the ART classifier on adversarial test examples
# plt.figure()
# plt.imshow(x_test_adv.numpy().transpose((0, 2, 3, 1))[0], cmap='gray')
# plt.show()

# get classifier predictions
model.eval()  # put in evaluation mode
with torch.no_grad():
    x_test_adv = x_test_adv.to(device)
    predictions = model(x_test_adv).to(torch.device('cpu'))

# transform net output into probabilities vector
predictions = scipy.special.softmax(predictions, axis=1)

# compute accuracy on the adversarial test set
accuracy = torch.sum(torch.argmax(predictions, axis=1) == y_test) / float(len(y_test))
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

# List of calibration methods to be compared
methods = {
    'None': methods.No_Calibration,
    'SC': methods.Non_Conformity_Score_Calibration,
    # 'CV+': arc.methods.CVPlus,
    # 'JK+': arc.methods.JackknifePlus,
    'HCC': methods.Non_Conformity_Score_Calibration,
    # 'CQC': arc.others.CQC,
    # 'HCC_LB': methods.Test_Score_Lower_Bound_Calibration,
    # 'SC_LB': methods.Test_Score_Lower_Bound_Calibration
    # 'HCC_UB': methods.Upper_Bound_Score_Calibration,
    # 'SC_UB': methods.Upper_Bound_Score_Calibration,
    'SC_Smooth': methods.Smoothed_Score_Calibration,
    'HCC_Smooth': methods.Smoothed_Score_Calibration
}

# create dataframe for storing results
results = pd.DataFrame()
for experiment in tqdm(range(n_experiments)):

    # Split test data into calibration and test
    x_calib, x_test_new, y_calib, y_test_new = train_test_split(x_test, y_test, test_size=0.5)

    # save sizes of calibration and test sets
    n_calib = x_calib.size()[0]
    n_test = x_test_new.size()[0]

    # Generate adversarial test examples
    x_test_adv = torch.from_numpy(attack.generate(x=x_test_new.numpy()))

    for method_name in methods:
        # Calibrate model with calibration method
        if method_name == "HCC_LB" or method_name == "HCC_UB" or method_name == "HCC_Smooth":
            method = methods[method_name](x_calib, y_calib, model, alpha, epsilon=epsilon,
                                          score_func=scores.class_probability_score)
        elif method_name == "SC_LB" or method_name == "SC_UB" or method_name == "SC_Smooth":
            method = methods[method_name](x_calib, y_calib, model, alpha, epsilon=epsilon,
                                          score_func=scores.generelized_inverse_quantile_score)
        elif method_name == "SC":
            method = methods[method_name](x_calib, y_calib, model, alpha,
                                          score_func=scores.generelized_inverse_quantile_score)
        elif method_name == "HCC":
            method = methods[method_name](x_calib, y_calib, model, alpha,
                                          score_func=scores.class_probability_score)
        else:
            method = methods[method_name](model, alpha)

        # Form prediction sets for test points on clean samples
        S = method.predict(x_test_new)

        # Evaluate results on clean examples
        res = scores.evaluate_predictions(S, x_test_new.numpy(), y_test_new.numpy(), conditional=False)

        res['Method'] = method_name
        res['Nominal'] = 1 - alpha
        res['n_test'] = n_test
        res['n_calib'] = n_calib
        res['n_train'] = n_train
        res['noise_norm'] = 0
        res['Black box'] = 'CNN'

        # Add results to the list
        results = results.append(res)

        # Form prediction sets for adversarial examples
        S = method.predict(x_test_adv)

        # Evaluate results on adversarial examples
        res = scores.evaluate_predictions(S, x_test_new.numpy(), y_test_new.numpy(), conditional=False)

        res['Method'] = method_name
        res['Nominal'] = 1 - alpha
        res['n_test'] = n_test
        res['n_calib'] = n_calib
        res['n_train'] = n_train
        res['noise_norm'] = epsilon
        res['Black box'] = 'CNN'

        # Add results to the list
        results = results.append(res)

# save results
results.to_csv('results.csv')

# plot results
# plot marginal coverage results
ax = sns.catplot(x="Black box", y="Coverage",
                 hue="Method", col="noise_norm",
                 data=results, kind="box",
                 height=4, aspect=.7)
# ax = sns.boxplot(y="Coverage", x="Black box", hue="Method", data=results)
for i, graph in enumerate(ax.axes[0]):
    graph.set(xlabel='Classifier', ylabel='Marginal coverage')
    graph.axhline(1 - alpha, ls='--', color="red")

ax.savefig("Marginal2.png")

# plot conditional coverage results
ax = sns.catplot(x="Black box", y="Conditional coverage",
                 hue="Method", col="noise_norm",
                 data=results, kind="box",
                 height=4, aspect=.7)
# ax = sns.boxplot(y="Conditional coverage", x="Black box", hue="Method", data=results)
for i, graph in enumerate(ax.axes[0]):
    graph.set(xlabel='Classifier', ylabel='Conditional coverage')
    graph.axhline(1 - alpha, ls='--', color="red")

ax.savefig("Conditional.png")

# plot interval size results
ax = sns.catplot(x="Black box", y="Size",
                 hue="Method", col="noise_norm",
                 data=results, kind="box",
                 height=4, aspect=.7)
# ax = sns.boxplot(y="Size cover", x="Black box", hue="Method", data=results)
for i, graph in enumerate(ax.axes[0]):
    graph.set(xlabel='Classifier', ylabel='Set Size')

ax.savefig("Size.png")
