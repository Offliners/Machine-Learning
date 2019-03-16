## Sales predict according to the area of shop and distance from MRT station
在古亭捷運站附近要開一家新飲料店，目前有其他分店的資訊，如下表所示:

|-|-|-|-|-|-|-|-|-|-|-|
|店面積(坪)|10|8|8|5|7|8|7|9|6|9|
|距離(m)|80|0|200|200|300|230|40|0|330|180|
|月營收(萬元)|46.9|36.6|37.1|20.8|24.6|29.7|36.6|43.6|19.8|36.4|

```python
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
```

## Result
迴歸係數 :  `4.12351586 -0.03452946`

截距 :  `6.845523384392724`

預測結果 : `店10坪且距離古亭捷運站100m，預估營業額44.62773616萬元`
