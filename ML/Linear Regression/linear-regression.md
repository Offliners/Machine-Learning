# Linear Regression
Take simple linear regression for example

Data :
```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y);
plt.show()
```
![data](https://github.com/Offliners/Machine-Learning/blob/master/ML/Linear%20Regression/data.png)
* Step 1. Data Preprocessing

* Step 2. Choose a class of model
```python
from sklearn.linear_model import LinearRegression
```
* Step 3.  Choose model hyperparameters
```python
model = LinearRegression(fit_intercept=True)
```
* Step 4.  Arrange data into a features matrix and target vector
```python
X = x[:, np.newaxis]
```
* Step 5. Fit the model to your data
```python
model.fit(X, y)
```
* Step 6. Predict labels for unknown data
```python
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
```
* Step 7. Visualization
```python
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
```
![model](https://github.com/Offliners/Machine-Learning/blob/master/ML/Linear%20Regression/model.png)
## Mathematical Model
* [Simple Linear Regression](Simple%20Linear%20Regression/Simple-Linear-Regression.md)
* [Polynomial Basic Functions](Polynomial%20Basic%20Functions/Polynomial-Basic-Functions.md)
* [Gaussian Basic Functions](Gaussian%20Basic%20Functions/gaussian_basic_functions.md)

## Optimization
* [Gradient Descent(GD)](Function/GD.py)

## Problems and Corrections
* Problem
  * [Overfitting](Overfitting/overfitting.md)
  
* Corrections
  * [Ridge Regularization(L2 Regularization)](Ridge%20Regression/ridge-regression.md)
  * [Lasso Regularization(L1 Regularization)](Lasso%20Regression/lasso-regression.md)
