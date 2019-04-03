# Boston
### Show Data
* Head Data
```python
import pandas as pd
df = pd.read_csv("Data/housing.data.txt",sep="\s+")
df.columns = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATTO","B","LSTAT","MEDV"]
print(df.head())
```
```shell
      CRIM   ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  PTRATTO       B  LSTAT  MEDV
0  0.02731  0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242.0     17.8  396.90   9.14  21.6
1  0.02729  0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242.0     17.8  392.83   4.03  34.7
2  0.03237  0.0   2.18     0  0.458  6.998  45.8  6.0622    3  222.0     18.7  394.63   2.94  33.4
3  0.06905  0.0   2.18     0  0.458  7.147  54.2  6.0622    3  222.0     18.7  396.90   5.33  36.2
4  0.02985  0.0   2.18     0  0.458  6.430  58.7  6.0622    3  222.0     18.7  394.12   5.21  28.7
```

* Scatter Matrix
```python
import matplotlib.pyplot as plt
cols = ["LSTAT","INDUS","NOX","RM","MEDV"]
sns.pairplot(df[cols],size = 2.5)
plt.tight_layout()
plt.show()
```
![Scatter]

* Heat Map
```python
import seaborn as sns
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt=".2f",annot_kws={"size":15},yticklabels=cols,xticklabels=cols)
plt.show()
```
![heat-map]

### Model
```python
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
```
* Loss Function
```python
sns.reset_orig()
plt.plot(range(1,lr.n_iter+1),lr.cost_)
plt.ylabel("SSE")
plt.xlabel("Epoch")
plt.show()
```
![LossFunction]

* Visualize Result
```python
plt.scatter(X_std,y_std,c="steelblue",edgecolors="white",s=70)
plt.plot(X_std,lr.predict(X_std),color="black",lw=2)
plt.xlabel("Average number of rooms[RM](Standarized)")
plt.ylabel("Price in $1000s [MEDV](Standarized)")
plt.show()
```
![Model]
