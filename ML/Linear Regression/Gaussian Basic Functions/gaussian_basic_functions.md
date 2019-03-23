## Gaussian Basic Functions
Main
```python 
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import GaussianFeatures as GF 
import numpy as np
import matplotlib.pyplot as plt

x = np.array([2,3,4])
poly = PolynomialFeatures(3,include_bias=False)
poly.fit_transform(x[:, None])
poly_model = make_pipeline(PolynomialFeatures(7),LinearRegression())
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)
poly_model.fit(x[:, np.newaxis],y)

gauss_model = make_pipeline(GF.GaussianFeatures(20),LinearRegression())
gauss_model.fit(x[:,np.newaxis],y)
xfit = np.linspace(0,10,1000)
yfit = gauss_model.predict(xfit[:,np.newaxis])

plt.scatter(x,y)
plt.plot(xfit,yfit)
plt.xlim(0,10)
plt.show()
```

Gaussian Module
```python
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

class GaussianFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
      
    @staticmethod 
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
        
    def fit(self, X, y=None):
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,self.width_, axis=1)
 ```
 ## Model
 ![Gauss model](https://github.com/Offliners/Machine-Learning/blob/master/ML/Linear%20Regression/Gaussian%20Basic%20Functions/gaussian_basic_functions.png)
