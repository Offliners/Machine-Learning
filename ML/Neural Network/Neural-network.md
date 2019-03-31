# Neural Network(NN)
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
