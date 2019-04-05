# Iris
其數據集包含了150個樣本，都屬於鳶尾屬下的三個亞屬，分別是山鳶尾、變色鳶尾和維吉尼亞鳶尾。

四個特徵被用作樣本的定量分析，它們分別是花萼和花瓣的長度和寬度。

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
* Misclassificantion Perceptron [(code link)](iris_perceptron.py)

![Misclassification](https://github.com/Offliners/Machine-Learning/blob/master/ML/Perceptron/Iris/misclassification.png)

* Plot Decision Regions [(code link)](iris_plot_decision_regions.py)

![decision_regions](https://github.com/Offliners/Machine-Learning/blob/master/ML/Perceptron/Iris/classifier.png)

* Adaline Learning Rate Comparison [(code link)](iris_adaline_comparison.py)

![Comparison](https://github.com/Offliners/Machine-Learning/blob/master/ML/Perceptron/Iris/Comparison.png)

* Improve Gradient Descent with Standardization [(code link)](iris_GD_standardization.py)

![standard_classification](https://github.com/Offliners/Machine-Learning/blob/master/ML/Perceptron/Iris/standard_classification.png)
![standard_loss](https://github.com/Offliners/Machine-Learning/blob/master/ML/Perceptron/Iris/standard_loss.png)

* Stochastic Gradient Descent [(code link)](iris_SGD.py)

![SGD_Classification]
![SGD_loss]

* Scikit Learn [(code link)](iris_sklearn.py)
![sklearn_classification]
