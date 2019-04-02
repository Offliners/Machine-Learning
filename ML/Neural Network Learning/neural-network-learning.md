# Neural Network Learning
## Loss Function
### Mean Squared Error
```python
import numpy as np

def mean_squared_error(y,t):
    return 0.5 * np.sum((y - t)**2)
```
* Test
```python
>>> t = [0,0,1,0,0,0,0,0,0,0] #Assume index-2 is answer
>>> y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0] #Example-1 : Probability of index-2 is the highest(0.6)
>>> mean_squared_error(np.array(y),np.array(t))
0.09750000000000003
>>> y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0] #Example-2 : Probability of index-7 is the highest(0.6)
>>> mean_squared_error(np.array(y),np.array(t))
0.5975
```

### Cross Entropy Error
```python
import numpy as np

def cross_entropy_error(y,t):
    delta = 1e-7 #Prevent from negative infinite
    return -np.sum(t * np.log(y + delta))
```
* Test
```python
>>> t = [0,0,1,0,0,0,0,0,0,0]
>>> y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
>>> cross_entropy_error(np.array(y),np.array(t))
0.510825457099338
>>> y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
>>> cross_entropy_error(np.array(y),np.array(t))
2.302584092994546
```

### Cross Entropy Error witch Batch
```python
import numpy as np
def cross_entropy_error(y,t):
    if y.ndim == 1:
	 t = t.reshape(1,t.size)
	 y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7))/batch_size
```

## Numerical Differentiation
### Differentiation
```python
def numerical_diff(f,x):
	h = 1e-4 #0.0001
	return (f(x+h)-f(x-h)) / (2 * h)
```
* Example
```python
def function_1(x):
    return 0.01 * x**2 + 0.1 * x

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0,20.0,0.1) #The range of array from 0 to 20 per 0.1 unit
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()
```
![function-1]
```python
>>> numerical_diff(function_1,5)
0.1999999999990898
>>> numerical_diff(function_1,10)
0.2999999999986347
```

### Gradient
