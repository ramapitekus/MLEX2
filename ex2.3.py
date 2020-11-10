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

    def get_probability(self, val, theta_mean, theta_variance):
        # TODO: returns the density value of the input value val
        # Note the input value val could be a vector rather than a single value
        # The code below is just for compilation. 
        # You need to replace it by your own code.
        k = norm.pdf(val, loc=theta_mean, scale=theta_variance**0.5)
        return k


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
        prob = np.count_nonzero(X == 1) / occurences
        if prob == 1 or prob == 0: prob = (prob+10**-6)/(occurences + (10**-6)*2) # additive smoothing

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
        return val*p
        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################


# Distribution for categorical features
class CatFeatureParam:
   # def __init__(self):
      #  self.unique_cat = np.unique(X)

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
        prob = np.zeros(len(unique_cat))

      #  for i in range((len(unique_cat))):
 #           print(np.count_nonzero(X == unique_cat))
          #  prob[i] = np. / occurences

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
        for i in range(len(self.unique_cat)):
            np.where(val == self.unique_cat[i], val, p[i]*val)
        return val
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



        if 'r' in feature_types:
            self.theta_mean = np.zeros((self.num_classes, self.feature_types.count('r'))) #mean in
            self.theta_variance = np.zeros((self.num_classes, np.count_nonzero(self.feature_types == 'r'))) #variance if
            self.cont = ContFeatureParam()

        if 'b' in feature_types:
            self.theta_bin = np.zeros((self.num_classes, np.count_nonzero(self.total_features == 'b'))) # prob matrix for 0,1 features
            self.bin = BinFeatureParam()
#zmenit vsetky a nechat len jednu thetu
        if 'c' in feature_types:
            self.theta_cat = np.zeros((self.num_classes, np.count_nonzero(self.feature_types == 'c'))) #prob matrix for multinoulli features
            self.cat = CatFeatureParam()


     #   if 'r' in feature_types:
     #       self.theta_mean = np.zeros((self.num_classes, self.feature_types.count('r'))) #mean in case of real, prob in case of bern/multi
     #       self.theta_variance = np.zeros((self.num_classes, self.feature_types.count('r'))) #variance if real, 0 otherwise
     #       self.cont = ContFeatureParam()
#
     #   if 'b' in feature_types:
     #       self.theta_bin = np.zeros((self.num_classes, self.feature_types.count('b'))) # prob matrix for 0,1 features
     #       self.bin = BinFeatureParam()
#
     #   if 'c' in feature_types:
     #       self.theta_cat = np.zeros((self.num_classes, self.feature_types.count('c'))) #prob matrix for multinoulli features
     #       self.cat = CatFeatureParam()


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
   #     print(np.unique(X[:, 0]))
        for i in range(self.total_features):
            if np.unique(X[:, i]).size > max:
                f = np.unique(X[:, i]).size
                max = np.unique(X[:, i]).size

        self.theta = np.zeros((self.num_classes, self.total_features, max)) ## opravit
   #     print(self.theta_cat.shape)
        for classnr in range(self.num_classes):
           self.pi.append(y[y == classnr].size / y.size) #probability of class y
           for feature in range(self.total_features):
               obs = np.array(X[y == self.uclass[classnr], feature]) # observations in X for specific class and feature
               if self.feature_types[feature] == 'r':
                   self.theta_mean[classnr, feature], self.theta_variance[classnr, feature] = self.cont.estimate(obs)
               if self.feature_types[feature] == 'b':
                   self.theta_bin[classnr, feature] = self.bin.estimate(obs)
               if self.feature_types[feature] == 'c':
                   prob = self.cat.estimate(obs)
                   self.theta_cat[classnr, feature, range(len(prob))] = prob
                   print(self.theta_cat[classnr, feature])


        return self


        ###################################################
        ##### YOUR CODE ENDS HERE #########################
        ###################################################

    # The function takes the data X as input, and predicts the class for the data
    def predict(self, X):

        numerator = np.ones((X.shape[0], self.num_classes)) * np.log(self.pi)

        for Class in range(self.num_classes):
            for feature in range(self.total_features):
                if self.feature_types[feature] == 'r':
                    numerator[:, Class] += np.log(self.cont.get_probability(X[:,feature], self.theta_mean[Class][feature],
                                                                            self.theta_variance[Class][feature]))
                if self.feature_types[feature] == 'b':
                    numerator[:, Class] += np.log(self.bin.get_probability(X[:, feature], self.theta_mean[Class, feature]))
                if self.feature_types[feature] == 'c':
                    numerator[:, Class] += np.log(self.cat.get_probability(X[:, feature], self.theta_mean[Class, feature]))

        denominator = logsumexp(numerator, axis=1)
        return self.uclass[np.argmax(numerator-denominator[:, np.newaxis], axis=1)]


#        ###################################################
#        ##### YOUR CODE ENDS HERE #########################
#        ###################################################

#from sklearn.datasets import load_iris
#
#iris = load_iris()
#print(type(iris))
#X, y = iris['data'], iris['target']
#
#N, D = X.shape
#Ntrain = int(0.8 * N)
#shuffler = np.random.permutation(N)
#Xtrain = X[shuffler[:Ntrain]]
#ytrain = y[shuffler[:Ntrain]]
#Xtest = X[shuffler[Ntrain:]]
#ytest = y[shuffler[Ntrain:]]
#
#nbc_iris = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
#nbc_iris.fit(Xtrain, ytrain)
#yhat = nbc_iris.predict(Xtest)
#test_accuracy = np.mean(yhat == ytest)
#print("Accuracy:", test_accuracy)

breast_cancer = pd.read_csv("breast-cancer.csv")
breast_cancer = pd.DataFrame(breast_cancer)
y =breast_cancer['Class']
X = breast_cancer.drop('Class', axis=1)

y = np.array(y)
X = np.array(X)

N, D = X.shape
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]

breast = NBC(feature_types=['c', 'c', 'c', 'c', 'b', 'c', 'b', 'c', 'b'], num_classes=2)
breast.fit(Xtrain, ytrain)
yhat = breast.predict(Xtest)
test_accuracy = np.mean(yhat == ytest)
print("Accuracy:", test_accuracy)