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
from torch.utils.data.dataset import random_split


from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import AutoProjectedGradientDescent
from art.attacks.evasion import ProjectedGradientDescent

from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

import pandas as pd
import Asaf.Score_Functions as scores
import Asaf.methods_new as methods
from sklearn.model_selection import train_test_split

# Define the neural network model,
# return logits instead of activation in forward method
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


alpha = 0.1
epsilon = 2
n_experiments = 5

# automatically choose device use gpu 0 if it is available o.w. use the cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print the choosen decive
print("device: ", device)

# Load the MNIST dataset using art function the data is nx28x28x1
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Swap axes to PyTorch's NCHW format nx1x28x28
x_train = torch.tensor(x_train.transpose((0, 3, 1, 2)).astype(np.float32)).to(device)
y_train = torch.tensor(y_train).to(device)
x_test = torch.tensor(x_test.transpose((0, 3, 1, 2)).astype(np.float32)).to(device)



# save the sizes of each on of the sets
n_train = np.shape(x_train)[0]
n_test = np.shape(x_test)[0]

# Create the model
model = Net().to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create the ART classifier
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

# Train the ART classifier
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

# Evaluate the ART classifier on  test examples

# get classifier predictions
predictions = classifier.predict(x_test)

# transform net output into probabilities vector
predictions = scipy.special.softmax(predictions, axis=1)

# compute accuracy on the test set
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=epsilon, norm=2)
# attack = AutoProjectedGradientDescent(estimator=classifier, eps=epsilon, norm=2)
# attack = ProjectedGradientDescent(estimator=classifier, eps=epsilon, norm=2)

x_test_adv = torch.tensor(attack.generate(x=x_test)).to(device)

# Step 7: Evaluate the ART classifier on adversarial test examples
#plt.figure()
#plt.imshow(x_test_adv.transpose((0, 2, 3, 1))[0],cmap='gray')
#plt.show()

predictions = classifier.predict(x_test_adv)

# transform net output into probabilities vector
predictions = scipy.special.softmax(predictions,axis=1)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))


# List of calibration methods to be compared
methods = {
     #'None': methods.No_Calibration,
     'SC': methods.Non_Conformity_Score_Calibration,
    # 'CV+': arc.methods.CVPlus,
    # 'JK+': arc.methods.JackknifePlus,
    # 'HCC': methods.Non_Conformity_Score_Calibration,
    # 'CQC': arc.others.CQC,
    #'HCC_LB': methods.Test_Score_Lower_Bound_Calibration,
    #'SC_LB': methods.Test_Score_Lower_Bound_Calibration
     #'HCC_UB': methods.Upper_Bound_Score_Calibration,
     #'SC_UB': methods.Upper_Bound_Score_Calibration,
    'SC_Smooth': methods.Smoothed_Score_Calibration
    # 'HCC_Smooth': methods.Smoothed_Score_Calibration
}

# create dataframe for storing results
results = pd.DataFrame()
for experiment in tqdm(range(n_experiments)):

    # Split test data into calibration and test
    x_calib, x_test_new, y_calib, y_test_new = train_test_split(x_test, y_test, test_size=0.8, random_state=(2020 + experiment))

    # save sizes of calibration and test sets
    n_calib = x_calib.size()[0]
    n_test = x_test_new.size()[0]

    # send data to device
    x_calib = x_calib.to(device)
    y_calib = torch.tensor(y_calib).to(device)
    x_test_new = x_test_new.to(device)

    # Generate adversarial test examples
    x_test_adv = torch.tensor(attack.generate(x=x_test_new)).to(device)

    for method_name in methods:
        # Calibrate model with calibration method
        if method_name == "HCC_LB" or method_name == "HCC_UB" or method_name == "HCC_Smooth":
            method = methods[method_name](x_calib, y_calib, classifier, alpha, epsilon=epsilon,
                                        score_func=scores.class_probability_score)
        elif method_name == "SC_LB" or method_name == "SC_UB" or method_name == "SC_Smooth":
            method = methods[method_name](x_calib, y_calib, classifier, alpha, epsilon=epsilon,
                                        score_func=scores.generelized_inverse_quantile_score)
        elif method_name == "SC":
            method = methods[method_name](x_calib, y_calib, classifier, alpha, score_func = scores.generelized_inverse_quantile_score)
        elif method_name == "HCC":
            method = methods[method_name](x_calib, y_calib, classifier, alpha, score_func = scores.class_probability_score)
        else:
            method = methods[method_name](classifier, alpha)

        # Form prediction sets for test points on clean samples
        S = method.predict(x_test_new)

        # Evaluate results on clean examples
        res = scores.evaluate_predictions(S, x_test_new, y_test_new,conditional=False)

        res['Method'] = method_name
        res['Nominal'] = 1 - alpha
        res['n_test'] = n_test
        res['n_calib'] = n_calib
        res['n_train'] = n_train
        res['noise_norm'] = 0
        res['Black box'] = 'CNN'

        # Add results to the list
        results = results.append(res)

        # Form prediction sets for dversarial exmaples
        S = method.predict(x_test_adv)

        # Evaluate results on adversarial exmaples
        res = scores.evaluate_predictions(S, x_test_new, y_test_new,conditional=False)

        res['Method'] = method_name
        res['Nominal'] = 1 - alpha
        res['n_test'] = n_test
        res['n_calib'] = n_calib
        res['n_train'] = n_train
        res['noise_norm'] = epsilon
        res['Black box'] = 'CNN'

        # Add results to the list
        results = results.append(res)

# plot marginal coverage results
ax = sns.catplot(x="Black box", y="Coverage",
                hue="Method", col="noise_norm",
                data=results, kind="box",
                height=4, aspect=.7)
# ax = sns.boxplot(y="Coverage", x="Black box", hue="Method", data=results)
for i, graph in enumerate(ax.axes[0]):
    graph.set(xlabel='Classifier', ylabel='Marginal coverage')
    graph.axhline(1 - alpha, ls='--', color="red")

ax.savefig("Marginal.png")


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