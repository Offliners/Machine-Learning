# Gaussian Naive Bayes
```python 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB

X,y = make_blobs(100,2,centers=2,random_state=2,cluster_std=1.5)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="RdBu")
plt.show()

model = GaussianNB()
model.fit(X,y)

rng = np.random.RandomState(0)
Xnew = [-6,-14] + [14,18]*rng.rand(2000,2)
ynew = model.predict(Xnew)

plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="RdBu")
lim = plt.axis()
plt.scatter(Xnew[:,0],Xnew[:,1],c=ynew,s=20,cmap="RdBu",alpha=0.1)
plt.axis(lim)
plt.show()

yprob = model.predict_proba(Xnew)
print(yprob[-8:].round(2))
```
## Data
![data](https://github.com/Offliners/Machine-Learning/blob/master/ML/Naive%20Bayes%20Classification/Gaussian%20Naive%20Bayes/data.png)

## Model
![model](https://github.com/Offliners/Machine-Learning/blob/master/ML/Naive%20Bayes%20Classification/Gaussian%20Naive%20Bayes/model.png)
