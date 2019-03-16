import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

areas_dists = np.array([[10,80],[8,0],[8,200],[5,200],[7,300],[8,230],[7,40],[9,0],[6,330],[9,180]])
sales = np.array([46.9,36.6,37.1,20.8,24.6,29.7,36.6,43.6,19.8,36.4])

X = pd.DataFrame(areas_dists, columns = ["Area","Distance"])
target = pd.DataFrame(sales,columns = ["Sale"])
y = target["Sale"]

lm = LinearRegression()
lm.fit(X,y)
print("迴歸係數 : ",lm.coef_)
print("截距 : ",lm.intercept_)

pre_area_dist = pd.DataFrame(np.array([[10,100]]))
pre_sale = lm.predict(pre_area_dist)
print(pre_sale)