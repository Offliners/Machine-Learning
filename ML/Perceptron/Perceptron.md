# Perceptron
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

### Perceptron
```python
import numpy as np

class Perceptron(object):
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.errors_ = []
        for i in range(self.n_iter):
            errors = 0
            for xi,target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0, 1 , -1)
```        

### Adative Linear Neuron(Adaline)

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

# Example
* [Iris](Iris/Iris.md)
