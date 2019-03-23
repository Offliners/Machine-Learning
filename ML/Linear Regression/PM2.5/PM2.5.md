## PM2.5 Predict
利用 linear regression或其他方法預測 PM2.5的數值

## Data Introduction
train.csv : 每個月前20天每個小時的氣象資料(每小時有18種測資)。共12個月。

test.csv : 排除train.csv中剩餘的資料，取連續9小時的資料當feature，預測第10小時的PM2.5值。總共取240筆不重複的test data。
