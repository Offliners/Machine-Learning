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
import numpy as np
import seaborn as sns
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt=".2f",annot_kws={"size":15},yticklabels=cols,xticklabels=cols)
plt.show()
```
![heat-map]

### Method
* Gradient Decent
![LossFunction]
![Model]