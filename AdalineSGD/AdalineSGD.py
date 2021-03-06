# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.colors import ListedColormap
from numpy.random import seed
import matplotlib.pyplot as plt

class AdalineSGD:
    """ADAptive Linear NEuron classifier.
    
    Parameters
    -------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    
    Attributes
    -------------
    w_ ： ld-array
        Weights after fitting.
    errors_ : list
        Number of misclassificaions in every epoch.
    shuffle : bool (default: True)
        Shuffles trainning data every epoch
        if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling
        and initializing the weights.
    """
    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
        
    def fit(self, x, y):
        """Fit training data.
        
        Parameters
        ------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples
            is the number of sampes and 
            n_features is the number of features.
        Y : array-like, shape = {n_samples}
            Target values.
            
        Returns
        ------------
        self : object
        
        """
        self._initialize_weights(x.shape[1])
        self.cost_ = []        
        for i in range(self.n_iter):
            if self.shuffle:
                x, y = self._shuffle(x, y)
            cost = []
            for xi, target in zip(x, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, x, y):
        """fit training data without reinitializing the weight"""
        if not self.w_initialized:
            self._initialize_weight(x.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        else:
            self._upadte_weights(xi, y)
        return self
    
    def _shuffle(self, x, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return x[r], y[r]
    
    def _initialize_weights(self, m):
        """Initailize weights to zeros"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input (self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
def plot_decision_regions(x, y, classifier, resolution=0.02):
    
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
        
    #plot the decision surface
    x1_min, x1_max = x[:, 0].min() -1, x[:,0].max() + 1
    x2_min, x2_max = x[:, 1].min() -1, x[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
        
    #plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[ y == cl, 0], y=x[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
# -*- coding: utf-8 -*-

