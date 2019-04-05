# Iris
[Data Link](Data/iris.data)
## Show Data
* Tail Data
```python
import pandas as pd

df = pd.read_csv("Data\iris.data",header=None)
print(df.tail())
```
```shell
       0    1    2    3               4
145  6.7  3.0  5.2  2.3  Iris-virginica
146  6.3  2.5  5.0  1.9  Iris-virginica
147  6.5  3.0  5.2  2.0  Iris-virginica
148  6.2  3.4  5.4  2.3  Iris-virginica
149  5.9  3.0  5.1  1.8  Iris-virginica
```

* Plot
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Data\iris.data",header=None)

# Select setosa and versicolor
y = df.iloc[0:100,4].values
y = np.where(y == "Iris-setosa", -1 , 1)
X = df.iloc[0:100,[0,2]].values

# Plot data
plt.scatter(X[:50,0],X[:50,1],color="red",marker="o",label="setosa")
plt.scatter(X[50:100,0],X[50:100,1],color="blue",marker="x",label="versicolor")
plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
plt.legend(loc="upper left")
plt.show()
```
![ShowPlot](https://github.com/Offliners/Machine-Learning/blob/master/ML/Perceptron/Iris/showplot.png)

## Method
* Misclassificantion Perceptron
```python
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
```
![Misclassification](https://github.com/Offliners/Machine-Learning/blob/master/ML/Perceptron/Iris/misclassification.png)

* Plot Decision Regions
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Perceptron as ppn
from matplotlib.colors import ListedColormap

df = pd.read_csv("Data\iris.data",header=None)

# Select setosa and versicolor
y = df.iloc[0:100,4].values
y = np.where(y == "Iris-setosa", -1 , 1)
X = df.iloc[0:100,[0,2]].values

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

ppn = ppn.Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)
plot_decision_regions(X,y,classifier=ppn)
plt.xlabel("sepal length [cm]")
plt.ylabel("petal length [cm]")
plt.legend(loc="upper left")
plt.show()
```
![decision_regions](https://github.com/Offliners/Machine-Learning/blob/master/ML/Perceptron/Iris/classifier.png)
