# Neural Network Learning
## Loss Function
### Mean Squared Error
```python
import numpy as np

def mean_squared_error(y,t):
	  return 0.5 * np.sum((y - t)**2)
```
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

```
