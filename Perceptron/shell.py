# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 21:09:15 2017

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''读入样本'''
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
'''选取样本并构图'''
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0,2]].values
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
'''训练数据'''
ppn=Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,
marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

plot_decision_regions(x, y, classifier=ppn)
plt.xlabel('s')
plt.ylabel('m')
plt.legend(loc='upper left')
plt.show()
