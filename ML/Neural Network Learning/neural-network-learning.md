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
