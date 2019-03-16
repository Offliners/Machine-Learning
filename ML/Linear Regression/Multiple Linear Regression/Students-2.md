## Students' Weights predict according to heights and waistline
在台灣調查某10位大學生的腰圍、身高與體重，如下表所示:

|腰圍(cm)|67|68|70|65|80|85|78|79|95|89|
|-|-|-|-|-|-|-|-|-|-|-|
|身高(cm)|160|165|167|170|165|167|178|182|175|172|
|體重(kg)|50|60|65|65|70|75|80|85|90|81|

```python 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

waists_heights = np.array([[67,160],[68,165],[70,167],[65,170],[80,165],[85,167],[78,178],[79,182],[95,175],[89,172]])
weights = np.array([50,60,65,65,70,75,80,85,90,81])

X = pd.DataFrame(waists_heights, columns = ["Waist","Height"])
target = pd.DataFrame(weights, columns = ["Weight"])
y = target["Weight"]

lm = LinearRegression()
lm.fit(X,y)
print("迴歸係數 : ",lm.coef_)
print("截距 : ",lm.intercept_)

pre_waist_height = pd.DataFrame(np.array([[66,164],[82,172]]))
pre_weight = lm.predict(pre_waist_height)
print(pre_weight)
```

## Result
迴歸係數 :  `0.71013574 1.07794276`

截距 :  `-166.36459730650566`

預測結果 : `學生腰圍66cm且164cm，預估有57.28697457kg` `學生腰圍82cm且身高172cm，預估有77.2726885kg`
