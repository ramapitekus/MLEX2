import pylab
pylab.rcParams['figure.figsize'] = (10., 10.)
from sklearn import linear_model

import pickle as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
import pandas as pd
from sklearn.linear_model import LogisticRegression as logistic, LogisticRegression


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
        return mean, variance

        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################

    def get_probability(self, val, theta):
        # TODO: returns the density value of the input value val
        # Note the input value val could be a vector rather than a single value
        # The code below is just for compilation. 
        # You need to replace it by your own code.
        return norm.pdf(val, loc=theta[0], scale=(theta[1]+10**-9)**0.5)+10**-9


# Distribution for binary features
class BinFeatureParam:

    def estimate(self, X):
        # TODO: Estimate the parameters for the Bernoulli distribution
        # so that it best describes the input data X
        # The code below is just for compilation.
        # You need to replace it by your own code.
        ###################################################
        ##### YOUR CODE STARTS HERE #######################
        ###################################################
        occurences = X.shape[0]

        unique, counts = np.unique(X, return_counts=True)
        if len(unique) > 2:
           raise AssertionError("more than 2 categories in binary")

        prob = (counts[0]+1) / (occurences + 2)

        return prob
        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################

    def get_probability(self, val, p):
        # TODO: returns the density value of the input value val
        # The code below is just for compilation.
        # You need to replace it by your own code.
        ###################################################
        ##### YOUR CODE STARTS HERE #######################
        ###################################################
        probVector = np.where(val == 0, p, 1-p)
        return probVector
        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################


# Distribution for categorical features
class CatFeatureParam:

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

        for indx in range(len(unique_cat)):
            prob[indx] = (np.count_nonzero(X == unique_cat[indx])+1) / (occurences+len(unique_cat))
        return prob
        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################

    def get_probability(self, val, theta):
        # TODO: returns the density value of the input value val
        # The code below is just for compilation.
        # You need to replace it by your own code.
        ###################################################
        ##### YOUR CODE STARTS HERE #######################
        ###################################################
        prob = np.zeros(len(val))
        theta = theta[:np.where(theta == 0)[0][0]]
        for i in range(len(theta)):
            prob = np.where(val == i, theta[i], prob)
        prob = np.where(prob == 0, 10**-6, prob)
        return prob
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

        self.pi = [] # probability of class y
        self.feature_types = np.array(feature_types)
        self.total_features = len(feature_types)
        self.num_classes = num_classes
        self.uclass = np.unique(np.array(y)) # unique classes in dataset
        self.cont = ContFeatureParam()
        self.bin = BinFeatureParam()
        self.cat = CatFeatureParam()

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


        self.theta = np.zeros((self.num_classes, self.total_features, 20)) #Theta is 3 dimensional array which stores probabilities
                                                                            #for all possible data (cont, binary, multinoulli)


        for classnr in range(self.num_classes):
            self.pi.append(y[y == self.uclass[classnr]].size / y.size)
            for feature in range(self.total_features):
                obs = np.array(X[y == self.uclass[classnr], feature]) # observations in X for specific class and feature
                if self.feature_types[feature] == 'r':
                    self.theta[classnr, feature, 0:2] = self.cont.estimate(obs)
                if self.feature_types[feature] == 'b':
                    self.theta[classnr, feature, 0] = self.bin.estimate(obs)
                if self.feature_types[feature] == 'c':
                    prob = self.cat.estimate(obs)
                    self.theta[classnr, feature, range(len(prob))] = prob
        return self


        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################

    # The function takes the data X as input, and predicts the class for the data
    def predict(self, X):

        numerator = np.ones((X.shape[0], self.num_classes)) * np.log(self.pi)

        for Class in range(self.num_classes):       #bayes formula
            for feature in range(self.total_features):
                if self.feature_types[feature] == 'r':
                    numerator[:, Class] += np.log(self.cont.get_probability(X[:, feature], self.theta[Class, feature, 0:2]))
                if self.feature_types[feature] == 'b':
                    numerator[:, Class] += np.log(self.bin.get_probability(X[:, feature], self.theta[Class, feature, 0]))
                if self.feature_types[feature] == 'c':
                    numerator[:, Class] += np.log(self.cat.get_probability(X[:, feature], self.theta[Class, feature, :]))
        denominator = logsumexp(numerator, axis=1)

        self.pi.clear()
        return self.uclass[np.argmax(numerator - denominator[:, np.newaxis], axis=1)]






def compareNBCvsLR(nbc, multi, X, y, num_runs=200, num_splits=10):
    # The code below is just for compilation.
    # You need to replace it by your own code.
    ###################################################
    ##### YOUR CODE STARTS HERE #######################
    ###################################################
    tst_errs_nbc = np.zeros((num_splits,num_runs))
    tst_errs_multi = np.zeros((num_splits,num_runs))#create the error matrix
    N, D = X.shape
    ratio=np.linspace(0,1,num_splits+2)[1:][:-1]#drop the first and last value because they are '0 'and '1'
    Ntrain = [int(ratio[i] * N) for i in range(ratio.shape[0])]#create the ratios' array
    for k in range(num_runs):#run 'num_runs' times
        shuffler = np.random.permutation(N)
        for i in range(len(Ntrain)):
            Xtrain = X[shuffler[:Ntrain[i]]]
            ytrain = y[shuffler[:Ntrain[i]]]
            Xtest = X[shuffler[Ntrain[i]:]]
            ytest = y[shuffler[Ntrain[i]:]]#devide the train and test set
            nbc.fit(Xtrain, ytrain)
            multi.fit(Xtrain, ytrain)#tarin the models
            y_multi_predict = multi.predict(Xtest)
            y_nbc_predict = nbc.predict(Xtest)#get the predict outcome
            tst_errs_multi[i,k] = np.mean(y_multi_predict == ytest)
            tst_errs_nbc[i,k] = np.mean(y_nbc_predict == ytest)#get the Correct rate
    return tst_errs_nbc.mean(axis=1), tst_errs_multi.mean(axis=1)# return the mean of correct rate
    ###################################################
    ##### YOUR CODE ENDS HERE #########################
    ###################################################








from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris['data'], iris['target']

N, D = X.shape
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]


nbc_iris = NBC(feature_types=['r','r','r','r'], num_classes=3)
nbc_iris.fit(Xtrain, ytrain)
yhat = nbc_iris.predict(Xtest)
test_accuracy = np.mean(yhat == ytest)

print("Accuracy:", test_accuracy)

##############################################################

from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import MultinomialNB as multinoulli


car = pd.read_csv("car.csv")
car = pd.DataFrame(car)
y =car['acceptability']
X = car.drop('acceptability', axis=1)
ordinal_encoder = OrdinalEncoder()
X = ordinal_encoder.fit_transform(X).astype(int)
y = np.array(y)
N, D = X.shape


car = NBC(feature_types=['c', 'c', 'c', 'c', 'c', 'c'], num_classes=4)
multi = multinoulli()
tst_errs_nbc_car, tst_errs_sklearn_car = compareNBCvsLR(car, multi, X, y, num_runs=10)
print("\ncar dataset \n")
print(tst_errs_nbc_car)
print("\n")
print(tst_errs_sklearn_car)
print("\n breast cancer \n")

#########################################################
breast = pd.read_csv("breast-cancer.csv")
breast = pd.DataFrame(breast)
y = breast['Class']
X = breast.drop('Class', axis=1)
ordinal_encoder = OrdinalEncoder()
X = ordinal_encoder.fit_transform(X).astype(int)
y = np.array(y)

multi1 = multinoulli()
breast = NBC(feature_types=['c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c'], num_classes=2)

tst_errs_nbc_breast, tst_errs_sklearn_breast = compareNBCvsLR(breast, multi1, X, y, num_runs=10)
print(tst_errs_nbc_breast)
print("\n")
print(tst_errs_sklearn_breast)