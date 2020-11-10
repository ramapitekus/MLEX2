import pylab
pylab.rcParams['figure.figsize'] = (10., 10.)
from sklearn import linear_model

import pickle as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
import pandas as pd



# Distribution for continuous features
class ContFeatureParam:

    def estimate(self, X):
        # TODO: Estimate the parameters for the Gaussian distribution 
        # so that it best describes the input data X
        # The code below is just for compilation. 
        # You need to replace it by your own code.
        ###################################################
        ##### YOUR CODE STARTS HERE #######################
        ###################################################

        mean = X.mean()
        variance = X.var(ddof=0)
        if variance == 0: variance += 10**-6

        return mean, variance

        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################

    def get_probability(self, val, theta):
        # TODO: returns the density value of the input value val
        # Note the input value val could be a vector rather than a single value
        # The code below is just for compilation. 
        # You need to replace it by your own code.
        return norm.pdf(val, loc=theta[0], scale=theta[1]**0.5)


# Distribution for binary features
class BinFeatureParam:

    def __init__(self):
        self.bin_vals = []

    def estimate(self, X):

        # TODO: Estimate the parameters for the Bernoulli distribution 
        # so that it best describes the input data X
        # The code below is just for compilation. 
        # You need to replace it by your own code.
        ###################################################
        ##### YOUR CODE STARTS HERE #######################
        ###################################################
        dict = {}
        occurences = X.shape[0]


        unique, counts = np.unique(X, return_counts=True)
        if len(unique) > 2:
            p = np.argmin(counts)
            unique = np.delete(unique, p)
            counts = np.delete(counts, p)

        for indx, unique in enumerate(unique):
            dict[indx] = unique

        self.bin_vals.append(dict)

        prob = counts[0] / occurences
        if prob == 1 or prob == 0: prob = (prob+10**-6)/(occurences + (10**-6)*2) # additive smoothing

        return prob
        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################

    def get_probability(self, val, p, b):
        # TODO: returns the density value of the input value val
        # The code below is just for compilation. 
        # You need to replace it by your own code.
        ###################################################
        ##### YOUR CODE STARTS HERE #######################
        ###################################################
        prob = np.where(val == self.bin_vals[b][0], p, 1-p)
        return prob
        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################


# Distribution for categorical features
class CatFeatureParam:

    def __init__(self):
        self.cat_vals = []

    def estimate(self, X):

        # TODO: Estimate the parameters for the Multinoulli distribution 
        # so that it best describes the input data X
        # The code below is just for compilation. 
        # You need to replace it by your own code.
        ###################################################
        ##### YOUR CODE STARTS HERE #######################
        ###################################################
        occurences = X.shape[0]
        unique_cat = np.unique(X)
        prob = np.zeros(unique_cat.shape)
        dict = {}

        for indx, val in enumerate(unique_cat):
            prob[indx] = np.count_nonzero(X == unique_cat[indx]) / occurences
            dict[indx] = val
        self.cat_vals.append(dict)

        return prob
        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################

    def get_probability(self, val, theta, cat_num):
        # TODO: returns the density value of the input value val
        # The code below is just for compilation. 
        # You need to replace it by your own code.
        ###################################################
        ##### YOUR CODE STARTS HERE #######################
        ###################################################
        prob = np.ones(len(val))
        for i in range(len(theta)):
            prob *= np.where(val == self.cat_vals[cat_num][i], theta[i], 1) # if
        return np.where(prob == 1, 0, prob) ##if none of features matches the data, set probability to zero
        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################


class NBC:
    # Inputs:
    #   feature_types: the array of the types of the features, e.g., feature_types=['r', 'r', 'r', 'r']
    #   num_classes: number of classes of labels


    def __init__(self, feature_types=[], num_classes=0):


        # The code below is just for compilation.
        # You need to replace it by your own code.
        ###################################################
        ##### YOUR CODE STARTS HERE #######################
        ###################################################
        self.pi = []
        self.feature_types = np.array(feature_types)
        self.total_features = len(feature_types)
        self.num_classes = num_classes
        self.uclass = np.unique(np.array(y)) # unique classes in dataset

        self.cont = ContFeatureParam()
        self.bin = BinFeatureParam()
        self.cat = CatFeatureParam()
        self.cat_length = []



        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################

    # The function uses the input data to estimate all the parameters of the NBC
    # You should use the parameters based on the types of the features
    def fit(self, X, y):
        # The code below is just for compilation.
        # You need to replace it by your own code.
        ###################################################
        ##### YOUR CODE STARTS HERE #######################
        ###################################################
        X = np.array(X)

        max = 0
        for i in range(self.total_features):
            if np.unique(X[:, i]).size > max:
                max = np.unique(X[:, i]).size

        self.theta = np.zeros((self.num_classes, self.total_features, max)) ## opravit


        for classnr in range(self.num_classes):
           self.pi.append(y[y == self.uclass[classnr]].size / y.size) #probability of class y
           for feature in range(self.total_features):
               obs = np.array(X[y == self.uclass[classnr], feature]) # observations in X for specific class and feature
               if self.feature_types[feature] == 'r':
                   self.theta[classnr, feature, 0:2] = self.cont.estimate(obs)
               if self.feature_types[feature] == 'b':
                   self.theta[classnr, feature, 0] = self.bin.estimate(obs)
               if self.feature_types[feature] == 'c':
                   prob = self.cat.estimate(obs)
                   if classnr == 0:
                     self.cat_length.append(len(prob))
                   self.theta[classnr, feature, range(len(prob))] = prob
        return self


        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################

    # The function takes the data X as input, and predicts the class for the data
    def predict(self, X):

        numerator = np.ones((X.shape[0], self.num_classes)) * np.log(self.pi)

        for Class in range(self.num_classes):
            c, b = 0, 0
            for feature in range(self.total_features):
                if self.feature_types[feature] == 'r':
                    numerator[:, Class] += np.log(self.cont.get_probability(X[:,feature], self.theta[Class, feature, 0:2]))
                if self.feature_types[feature] == 'b':
                    numerator[:, Class] += np.log(self.bin.get_probability(X[:, feature], self.theta[Class, feature, 0], b))
                    b += 1
                if self.feature_types[feature] == 'c':
                    numerator[:, Class] += np.log(self.cat.get_probability(X[:, feature], self.theta[Class, feature, range(self.cat_length[c])], c))
                    c += 1

        denominator = logsumexp(numerator, axis=1)

        return self.uclass[np.argmax(numerator-denominator[:, np.newaxis], axis=1)]


#        ###################################################
#        ##### YOUR CODE ENDS HERE #########################
#        ###################################################

from sklearn.datasets import load_iris

iris = load_iris()
print(type(iris))
X, y = iris['data'], iris['target']

N, D = X.shape
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]

nbc_iris = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
nbc_iris.fit(Xtrain, ytrain)
yhat = nbc_iris.predict(Xtest)
test_accuracy = np.mean(yhat == ytest)
print("Accuracy:", test_accuracy)

#breast_cancer = pd.read_csv("breast-cancer.csv")
#breast_cancer = pd.DataFrame(breast_cancer)
#y =breast_cancer['Class']
#X = breast_cancer.drop('Class', axis=1)
#
#y = np.array(y)
#X = np.array(X)
#
#N, D = X.shape
#Ntrain = int(0.8 * N)
#shuffler = np.random.permutation(N)
#Xtrain = X[shuffler[:Ntrain]]
#ytrain = y[shuffler[:Ntrain]]
#Xtest = X[shuffler[Ntrain:]]
#ytest = y[shuffler[Ntrain:]]
#
#breast = NBC(feature_types=['c', 'c', 'c', 'c', 'b', 'c', 'b', 'c', 'b'], num_classes=2)
#breast.fit(Xtrain, ytrain)
#yhat = breast.predict(Xtest)
#test_accuracy = np.mean(yhat == ytest)
#print("Accuracy:", test_accuracy)