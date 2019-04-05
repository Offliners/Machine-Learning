import pandas as pd
from sklearn.preprocessing import StandardScaler 
import GD
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("Data/housing.data.txt",sep="\s+")
df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATTO","B","LSTAT","MEDV"]

# Linear Regression GD
X = df[["RM"]].values
y = df[["MEDV"]].values

sc_X = StandardScaler()
sc_y = StandardScaler()

X_std = sc_X.fit_transform(X)
y = np.array(y).reshape(-1,1)
y_std = sc_y.fit_transform(y)
y_std = y_std.flatten()
lr = GD.LinearRegressionGD()
lr.fit(X_std,y_std)

# Visualize
sns.reset_orig()
plt.plot(range(1,lr.n_iter+1),lr.cost_)
plt.ylabel("SSE")
plt.xlabel("Epoch")
plt.show()

plt.scatter(X_std,y_std,c="steelblue",edgecolors="white",s=70)
plt.plot(X_std,lr.predict(X_std),color="black",lw=2)
plt.xlabel("Average number of rooms[RM](Standarized)")
plt.ylabel("Price in $1000s [MEDV](Standarized)")
plt.show()
