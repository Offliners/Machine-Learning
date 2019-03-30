# Numpy
### Basic
```python
>>> import numpy as np
>>> x = np.array([1.0,2.0,3.0])
>>> print(x)
[1. 2. 3.]
>>> type(x)
<class 'numpy.ndarray'>
```

### Arithmetic operation of Numpy
```python
>>> x = np.array([1.0,2.0,3.0])
>>> y = np.array([2.0,4.0,6.0])
>>> x + y #Element-wise add
array([3., 6., 9.])
>>> x - y #Element-wise subtract
array([-1., -2., -3.])
>>> x * y #Element-wise product
array([ 2.,  8., 18.])
>>> x/y #Element-wise divide
array([0.5, 0.5, 0.5])
>>> x / 2.0 #Broadcast
array([0.5, 1. , 1.5])
```

### N-dimensional Array
```python
>>> A = np.array([[1,2],[3,4]])
>>> print(A)
[[1 2]
 [3 4]]
>>> A.shape
(2, 2)
>>> A.dtype
dtype('int32')
>>> B = np.array([[3,0],[0,6]])
>>> A + B
array([[ 4,  2],
       [ 3, 10]])
>>> A * B
array([[ 3,  0],
       [ 0, 24]])
>>> A * 10
array([[10, 20],
       [30, 40]])
```
