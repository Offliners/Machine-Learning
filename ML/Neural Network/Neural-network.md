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
![step_function]

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
![sigmoid]

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
![ReLU]
