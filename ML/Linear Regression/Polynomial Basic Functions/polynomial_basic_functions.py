import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

x = np.array([2,3,4])
poly = PolynomialFeatures(3,include_bias=False)
poly.fit_transform(x[:, None])

poly_model = make_pipeline(PolynomialFeatures(7),LinearRegression())
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

poly_model.fit(x[:, np.newaxis],y)
xfit = np.linspace(0,10,1000)
yfit = poly_model.predict(xfit[:,np.newaxis])

plt.scatter(x,y)
plt.plot(xfit,yfit)
plt.show()