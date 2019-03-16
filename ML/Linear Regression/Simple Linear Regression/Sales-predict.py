import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

temp = np.array([29,28,34,31,25,29,32,31,24,33,25,31,26,30])
sales = np.array([7.7,6.2,9.3,8.4,5.9,6.4,8.0,7.5,5.8,9.1,5.1,7.3,6.5,8.4])

X = pd.DataFrame(temp,columns = ["Temperature"])

target = pd.DataFrame(sales,columns = ["Drink sales"])

y = target["Drink sales"]

lm = LinearRegression()
lm.fit(X,y)
print("迴歸係數 : ",lm.coef_)
print("截距 : ",lm.intercept_)

pre_temp = pd.DataFrame(np.array([26,30]))
pre_sales = lm.predict(pre_temp)
print(pre_sales)

plt.scatter(temp,sales)
regression_sales = lm.predict(X)
plt.plot(temp,regression_sales,color = "blue")
plt.plot(pre_temp,pre_sales,color = "red",marker = "o",markersize = 10)
plt.show()