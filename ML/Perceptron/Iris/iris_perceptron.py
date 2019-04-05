import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Perceptron as ppn

df = pd.read_csv("Data\iris.data",header=None)

# Select setosa and versicolor
y = df.iloc[0:100,4].values
y = np.where(y == "Iris-setosa", -1 , 1)
X = df.iloc[0:100,[0,2]].values

ppn = ppn.Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker="o")
plt.xlabel("Epoch")
plt.ylabel("Number of Misclassifications")
plt.show()
