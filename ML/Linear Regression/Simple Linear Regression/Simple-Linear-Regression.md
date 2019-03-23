## Simple Linear Regression
```python 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x,y)

model = LinearRegression(fit_intercept=True)
model.fit(x[:,np.newaxis],y)
xfit = np.linspace(0,10,1000)
yfit = model.predict(xfit[:,np.newaxis])
plt.scatter(x,y)
plt.plot(xfit,yfit)

print("Model scope : ",model.coef_)
print("Model intercept : ",model.intercept_)
plt.show()
```
## Data
![data](https://github.com/Offliners/Machine-Learning/blob/master/ML/Linear%20Regression/Simple%20Linear%20Regression/data.png)
## Model
![Simple-Linear-Regression](https://github.com/Offliners/Machine-Learning/blob/master/ML/Linear%20Regression/Simple%20Linear%20Regression/simple_linear_regression.png)
## Result
Model scope     :  2.02720881
Model intercept :  -4.9985770855532
