# MNIST
Use [`minst.py`](MINST/minst.py) to load MINST dataset

### MINST Datatype
```python
import sys,os
sys.path.append(os.pardir)
from mnist import load_mnist

(x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
```
* Result
```shell
(60000, 784)
(60000,)
(10000, 784)
(10000,)
```

### MINST Show
```python
import sys,os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False)
img = x_train[0]
label = t_train[0]
print(label)
print(img.shape)
img = img.reshape(28,28)
print(img.shape)
img_show(img)
```
* Result
```shwll
5
(784,)
(28, 28)
```
![MINST Show]

### MINST Neural Network
```python
import sys,os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image
from function import sigmoid,softmax
import pickle

def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl","rb") as f:
        network = pickle.load(f)
    return network

def predict(network,x):
    W1,W2,W3 = network["W1"],network["W2"],network["W3"]
    b1,b2,b3 = network["b1"],network["b2"],network["b3"]
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    return y

x,t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network,x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy : " + str(float(accuracy_cnt)/len(x)))
```
* Result
`Accuracy : 0.9352`

### Batch
```python
>>> list(range(0,10))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> list(range(0,10,3))
[0, 3, 6, 9]
>>> import numpy as np
>>> x = np.array([[0.1,0.8,0.1],[0.3,0.1,0.6],[0.2,0.5,0.3],[0.8,0.1,0.1]])
>>> y = np.argmax(x,axis = 1)
>>> print(y)
[1 2 1 0]
>>> y = np.array([1,2,1,0])
>>> t = np.array([1,2,0,0])
>>> print(y ==t)
[ True  True False  True]
>>> np.sum(y == t)
3
```

### MINST Batch processing
```python
import sys,os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image
from function import sigmoid,softmax
import pickle

def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl","rb") as f:
        network = pickle.load(f)
    return network

def predict(network,x):
    W1,W2,W3 = network["W1"],network["W2"],network["W3"]
    b1,b2,b3 = network["b1"],network["b2"],network["b3"]
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    return y

x,t = get_data()
network = init_network()
batch_size = 100 #Amount of batch
accuracy_cnt = 0
for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch,axis = 1) 
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy : " + str(float(accuracy_cnt)/len(x)))
```
* Result
`Accuracy : 0.9352`
