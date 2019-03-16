import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

heights = np.array([147.9,163.5,159.8,155.1,163.3,158.7,172.0,161.2,153.9,161.6])
weights = np.array([41.7,60.2,47.0,53.2,48.3,55.2,58.5,49.0,46.7,52.5])

x = pd.DataFrame(heights,columns = ["Height"])
target = pd.DataFrame(weights,columns = ["Weight"])
y = target["Weight"]

lm = LinearRegression()
lm.fit(x,y)

print("迴歸係數 : ",lm.coef_)
print("截距 : ",lm.intercept_)

pre_heights = pd.DataFrame(np.array([150,160,180]))
pre_weights = lm.predict(pre_heights)
print(pre_weights)

plt.scatter(heights,weights)
regression_weights =  lm.predict(x)
plt.plot(heights,regression_weights,color = "blue")
plt.plot(pre_heights,pre_weights,color = "red",marker = "o",markersize = 10)
plt.show()