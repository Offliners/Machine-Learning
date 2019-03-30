# Python Basic
### Arithmetic operation
```python
>>> 1 - 2
-1
>>> 4 * 5
20
>>> 7 / 5
1.4
>>> 3 ** 2
9
```

### Data Type
```python
>>> type(10)
<class 'int'>
>>> type(2.718)
<class 'float'>
>>> type("hello")
<class 'str'>
```

### Variable
```python
>>> x = 10 #Default value
>>> print(x) #Output x
10
>>> x = 100 #Change value
>>> print(x)
100
>>> y = 3.14
>>> x * y
314.0
>>> type(x * y)
<class 'float'>
```

### Set
```python
>>> a = [1,2,3,4,5] #Build set
>>> print(a) #Output set
[1, 2, 3, 4, 5]
>>> len(a) #Length of set
5
>>> a[0] #Get the first value
1
>>> a[4]
5
>>> a[4] = 99 #Change the fourth value
>>> print(a)
[1, 2, 3, 4, 99]
>>> a[0:2] #Get the elements from index-0 to index-2(Not included index-2)
[1, 2]
>>> a[1:] #Get the elements from index-1 to the last
[2, 3, 4, 99]
>>> a[:3] #Get the elements from the first to index-3(Not included index-3)
[1, 2, 3]
>>> a[:-1] #Get the elemetns from the first to the last two
[1, 2, 3, 4]
>>> a[:-2] #Get the elemetns from the first to the last three
[1, 2, 3]
```

### Dictionary
```python
>>> me = {"Height":160} #Build dictionart
>>> me["Height"] #Get value
160
>>> me["Height"] = 180 #Change value
>>> print(me)
{'Height': 180}
```

### Bool Type
```python
>>> hungry = True
>>> sleepy = False
>>> type(hungry)
<class 'bool'>
>>> not hungry
False
>>> hungry and sleepy
False
>>> hungry or sleepy
True
```

### If Statement
```python
>>> hungry = True
>>> if hungry:
	      print("I am hungry")

I am hungry
>>> hungry = False
>>> if hungry:
	      print("I am hungry")
    else:
        print("I am not hungry")
        print("I am sleepy")
I am not hungry
I am sleepy
```
