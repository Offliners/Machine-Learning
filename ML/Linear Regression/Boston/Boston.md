# Boston
這個房屋數據集包含506個樣本，其中包含14個特徵，簡單描述如下:

* `CRIM` : 某城鎮的人均犯罪率

* `ZN` : 超過25000平方呎的住宅用地區塊，所佔的比例

* `INDUS` : `某城鎮非(零售業)的商用用地比例(英畝)

* `CHAR` : 關於查爾斯河的虛擬變數(如果某區域以河道為界，該屬性質為1;否則為0)

* `NOX` : 一氧化氮濃度(以10ppm為單位)

* `RM` : 每戶平均有幾個房間

* `AGE` : 在1940年之前所建的房屋，屋主自用的比例

* `DIS` : 到波士頓五個就業服務站的(加權)距離

* `RAD` : 使用高速公路方便性的指數

* `TAX` : (總價 - 房屋稅)的比例(單位 : 10000美金)

* `PTRATIO` : 某城鎮的師生比

* `B` : 以1000(Bk - 0.63)^2 ，Bk是非洲裔的比例

* `LSTAT` : 低社經地位的人口比例

* `MEDV` : 自用住宅的房價中位數(單位 : 1000美金)

## Show Data
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
![Scatter](https://github.com/Offliners/Machine-Learning/blob/master/ML/Linear%20Regression/Boston/Scatter.png)

* Heat Map
```python
import numpy as np
import seaborn as sns
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt=".2f",annot_kws={"size":15},yticklabels=cols,xticklabels=cols)
plt.show()
```
![heat-map](https://github.com/Offliners/Machine-Learning/blob/master/ML/Linear%20Regression/Boston/Heat-Map.png)

## Method
* Gradient Decent  [(code link)](Boston_GD.py)

![LossFunction](https://github.com/Offliners/Machine-Learning/blob/master/ML/Linear%20Regression/Boston/Boston-Loss.png)
![Model](https://github.com/Offliners/Machine-Learning/blob/master/ML/Linear%20Regression/Boston/model.png)

* Scikit Learn
