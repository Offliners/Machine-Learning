# Perceptron
### And Gate
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
