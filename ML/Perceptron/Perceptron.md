<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# Perceptron
$$\begin{Bmatrix}1 & 2\\\\3 &4\end{Bmatrix}$$
### And Gate
|x1|x2|y|
|-|-|-|
|0|0|0|
|0|1|0|
|1|0|0|
|1|1|1|
```python
def And(x1,x2):
    w1,w2,theta = 0.5,0.5,0.7
    tmp = w1 * x1 + w2 * x2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

print(And(0,0)) #Output 0
print(And(0,1)) #Output 0
print(And(1,0)) #Output 0
print(And(1,1)) #Output 1
```

### Use Weight and Bias
```python
>>> import numpy as np
>>> x = np.array([0,1]) #Input
>>> w = np.array([0.5,0.5]) #Weight
>>> b = -0.7 #Bias
>>> w*x
array([0. , 0.5])
>>> np.sum(w*x)
0.5
>>> np.sum(w*x) + b
-0.19999999999999996  //About -2
```

### Combine And Gate with Weights and Bias
```python
import numpy as np
def And(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp =  np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(And(0,0)) #Output 0
print(And(0,1)) #Output 0
print(And(1,0)) #Output 0
print(And(1,1)) #Output 1
```

### NAND Gate
|x1|x2|y|
|-|-|-|
|0|0|1|
|0|1|1|
|1|0|1|
|1|1|0|
```python
import numpy as np
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5]) #Only weight and bias different from And gate
    b = 0.7
    tmp =  np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(NAND(0,0)) #Output 1
print(NAND(0,1)) #Output 1
print(NAND(1,0)) #Output 1
print(NAND(1,1)) #Output 0
```

### Or Gate
|x1|x2|y|
|-|-|-|
|0|0|0|
|0|1|1|
|1|0|1|
|1|1|1|
```python
import numpy as np
def Or(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5]) #Only weight and bias different from And gate
    b = -0.2
    tmp =  np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print(Or(0,0)) #Output 0
print(Or(0,1)) #Output 1
print(Or(1,0)) #Output 1
print(Or(1,1)) #Output 1
```

# Multi-Layered Perceptron
### XOR Gate
|x1|x2|s1|s2|y|
|-|-|-|-|-|
|0|0|1|0|0|
|0|1|1|1|1|
|1|0|1|1|1|
|1|1|0|1|1|
```python
def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = Or(x1,x2)
    y = And(s1,s2)
    return y

print(XOR(0,0)) #Output 0
print(XOR(0,1)) #Output 1
print(XOR(1,0)) #Output 1
print(XOR(1,1)) #Output 0
```
