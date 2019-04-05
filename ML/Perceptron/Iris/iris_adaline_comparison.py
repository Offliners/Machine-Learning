import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from AdalineGD import *

df = pd.read_csv("Data\iris.data",header=None)

# Select setosa and versicolor
y = df.iloc[0:100,4].values
y = np.where(y == "Iris-setosa", -1 , 1)
X = df.iloc[0:100,[0,2]].values

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
ada1 = AdalineGD(n_iter=10,eta=0.01).fit(X,y)
ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker="o")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("log(Sum-squared-error)")
ax[0].set_title("Adaline - Learning rate 0.01")

ada2 = AdalineGD(n_iter=10,eta=0.0001).fit(X,y)
ax[1].plot(range(1,len(ada1.cost_)+1),np.log10(ada2.cost_),marker="o")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("log(Sum-squared-error)")
ax[1].set_title("Adaline - Learning rate 0.0001")

plt.show()
