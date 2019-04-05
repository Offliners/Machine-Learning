# Iris

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
![Misclassification]
