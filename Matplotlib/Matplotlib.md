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

### The Function of pyplot
```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> x = np.arange(0,6,0.1)
>>> y1 = np.sin(x)
>>> y2 = np.cos(x)
>>> plt.plot(x,y1,label="sin")
[<matplotlib.lines.Line2D object at 0x000001BA02B825F8>]
>>> plt.plot(x,y2,linestyle="--",label="cos") #Draw with dotted line
[<matplotlib.lines.Line2D object at 0x000001BA7FDD85C0>]
>>> plt.xlabel("x") #Label of x-axis
Text(0.5, 0, 'x')
>>> plt.ylabel("y") ##Label of y-axis
Text(0, 0.5, 'y')
>>> plt.title("sin & cos") #Titel of chart
Text(0.5, 1.0, 'sin & cos')
>>> plt.legend()
<matplotlib.legend.Legend object at 0x000001BA02B82748>
>>> plt.show()
```
![sin&cos]

### Show Image
```python
import matplotlib.pyplot as plt
from matplotlib.image import imread
img = imread("NTNU.png")
plt.imshow(img)
plt.show()
```
![NTNU-show]
