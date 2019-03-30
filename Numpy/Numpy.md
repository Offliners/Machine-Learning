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

### Broadcast
```python
>>> A = np.array([[1,2],[3,4]])
>>> B = np.array([10,20])
>>> A * B
array([[10, 40],
       [30, 80]])
```

### Access elements
```python
>>> X = np.array([[51,55],[14,19],[0,4]])
>>> print(X)
[[51 55]
 [14 19]
 [ 0  4]]
>>> X[0] #The first row
array([51, 55])
>>> X[0][1] #Element(0,1)
55
>>> for row in X:
	     print(row)
	
[51 55]
[14 19]
[0 4]
>>> X = X.flatten() #Convert X to 1-dimensional array
>>> print(X)
[51 55 14 19  0  4]
>>> X[np.array([0,2,4])] #Get elements of index-0,2,4
array([51, 14,  0])
>>> X > 15
array([ True,  True, False,  True, False, False])
>>> X[X>15]
array([51, 55, 19])
```
