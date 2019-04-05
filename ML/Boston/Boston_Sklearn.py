import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Data/housing.data.txt",sep="\s+")
df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATTO","B","LSTAT","MEDV"]

X = df[["RM"]].values
y = df[["MEDV"]].values

slr = LinearRegression()
slr.fit(X,y)
print("Slope : %.3f" % slr.coef_[0])
print("Intercept : %.3f" % slr.intercept_)
plt.scatter(X,y,c="steelblue",edgecolors="white",s=70)
plt.plot(X,slr.predict(X),color="black",lw=2)
plt.xlabel("Average number of rooms[RM](Standarized)")
plt.ylabel("Price in $1000s [MEDV](Standarized)")
plt.show()
