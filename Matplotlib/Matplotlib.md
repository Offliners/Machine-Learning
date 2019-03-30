# Matplotlib Basic
### Draw Simple Chart
```python
>>> import matplotlib.pyplot as plt 
>>> import numpy as np
>>> x = np.arange(0,6,0.1) #Build data #Produce data  unit from 0 to 6 
>>> y = np.sin(x)
>>> plt.plot(x,y) #Draw chart
[<matplotlib.lines.Line2D object at 0x0000028B90413208>]
>>> plt.show()
```
![sin-image]
