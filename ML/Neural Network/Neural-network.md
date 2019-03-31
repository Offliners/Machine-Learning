# Neural Network(NN)
## Activation Function
### Step Function
```python
import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0,5.0,0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1) #Set range of y
plt.show()
```
![step_function](https://github.com/Offliners/Machine-Learning/blob/master/ML/Neural%20Network/Image/step_function.png)

### Sigmoid Function
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0,5.0,0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1) #Set range of y
plt.show()
```
![sigmoid](https://github.com/Offliners/Machine-Learning/blob/master/ML/Neural%20Network/Image/sigmoid.png)

### ReLU(Rectified Linear Unit) Function
```python
import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    return np.maximum(0,x)

x = np.arange(-5.0,5.0,0.1)
y = ReLU(x)
plt.plot(x,y)
plt.ylim(-1,5) #Set range of y
plt.show()
```
![ReLU](https://github.com/Offliners/Machine-Learning/blob/master/ML/Neural%20Network/Image/ReLU.png)

### N-Dimensional Array
```python
>>> import numpy as np
>>> A = np.array([1,2,3,4])
>>> print(A)
[1 2 3 4]
>>> np.ndim(A) #Get dimension of A
1
>>> A.shape #Get shape of A
(4,)
>>> A.shape[0]
4
>>> B = np.array([[1,2],[3,4],[5,6]])
>>> print(B)
[[1 2]
 [3 4]
 [5 6]]
>>> np.ndim(B)
2
>>> B.shape
(3, 2)
```

### Matrices Product
```python
>>> import numpy as np
>>> A = np.array([[1,2],[3,4]])
>>> A.shape
(2, 2)
>>> B = np.array([[5,6],[7,8]])
>>> B.shape
(2, 2)
>>> np.dot(A,B)
array([[19, 22],
       [43, 50]])
>>> A = np.array([[1,2,3],[4,5,6]])
>>> A.shape
(2, 3)
>>> B = np.array([[1,2],[3,4],[5,6]])
>>> B.shape
(3, 2)
>>> np.dot(A,B)
array([[22, 28],
       [49, 64]])
>>> C = np.array([[1,2],[3,4]])
>>> C.shape
(2, 2)
>>> A.shape
(2, 3)
>>> np.dot(A,C)
Traceback (most recent call last):
  File "<pyshell#24>", line 1, in <module>
    np.dot(A,C)
ValueError: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)
>>> A = np.array([[1,2],[3,4],[5,6]])
>>> A.shape
(3, 2)
>>> B = np.array([7,8])
>>> B.shape
(2,)
>>> np.dot(A,B)
array([23, 53, 83])
```

### Product of Neural Network
```python
>>> import numpy as np
>>> X = np.array([1,2])
>>> X.shape
(2,)
>>> W = np.array([[1,3,5],[2,4,6]])
>>> print(W)
[[1 3 5]
 [2 4 6]]
>>> W.shape
(2, 3)
>>> Y = np.dot(X,W)
>>> print(Y)
[ 5 11 17]
```

### Eexcute of Three-Layer Neural Network
```python
>>> import numpy as np
>>> def sigmoid(x):
	return 1 / (1 + np.exp(-x))

>>> X = np.array([1.0,0.5])
>>> W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])

>>> B1 = np.array([0.1,0.2,0.3])
>>> print(W1.shape)
(2, 3)
>>> print(X.shape)
(2,)
>>> print(X.shape)
(2,)
>>> A1 = np.dot(X,W1) + B1
>>> Z1 = sigmoid(A1)
>>> print(A1)
[0.3 0.7 1.1]
>>> print(Z1)
[0.57444252 0.66818777 0.75026011]
>>> W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
>>> B2 = np.array([0.1,0.2])
>>> print(Z1.shape)
(3,)
>>> print(W2.shape)
(3, 2)
>>> print(B2.shape)
(2,)
>>> A2 = np.dot(Z1,W2) + B2
>>> Z2 = sigmoid(A2)
>>> def identity_function(x):
	return x

>>> W3 = np.array([[0.1,0.3],[0.2,0.4]])
>>> B3 = np.array([0.1,0.2])
>>> A3 = np.dot(Z2,W3) + B3
>>> Y = identity_function(A3)
>>> print(Y)
[0.31682708 0.69627909]
```

### Unified execution processing
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
	return x

def init_network():
    network = {}
    network["W1"] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network["b1"] = np.array([0.1,0.2,0.3])
    network["W2"] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network["b2"] = np.array([0.1,0.2])
    network["W3"] = np.array([[0.1,0.3],[0.2,0.4]])
    network["b3"] = np.array([0.1,0.2])
    return network

def forward(network,x):
    W1,W2,W3 = network["W1"],network["W2"],network["W3"]
    b1,b2,b3 = network["b1"],network["b2"],network["b3"]
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y) #[0.31682708 0.69627909]
```
