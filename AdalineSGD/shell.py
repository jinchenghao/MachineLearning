# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0,2]].values

"""处理样本数据，使用标准化特征缩放"""
x_std = np.copy(x)
x_std[:,0] = (x[:,0] - x[:,0].mean()) / x[:,0].std()
x_std[:,1] = (x[:,1] - x[:,1].mean()) / x[:,1].std()

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(x_std, y)
plot_decision_regions(x_std, y, classifier=ada)
plt.title("Adaline - Stochastic Dradient Descent")
plt.xlabel("sepal length [standardized]")
plt.xlabel("pepal length [standardized]")
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Average Cost")
plt.show()

