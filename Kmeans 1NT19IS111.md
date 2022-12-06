```python
import pandas
ds = pandas.read_csv('kmeans.csv')
```


```python
import numpy as np
k = 3
c1 = [[6.2,3.2],[6.6,3.7],[6.5,3.0]]
c = np.array(c1)
print(c.shape)
type(c)
a = ds.to_numpy()
print(a)
```

    (3, 2)
    [[5.9 3.2]
     [4.6 2.9]
     [6.2 2.8]
     [4.7 3.2]
     [5.5 4.2]
     [5.  3. ]
     [4.9 3.1]
     [6.7 3.1]
     [5.1 3.8]
     [6.  3. ]]
    


```python
ctold = np.zeros(c.shape)
print(ctold)
```

    [[0. 0.]
     [0. 0.]
     [0. 0.]]
    


```python
op = np.zeros(10)
print(op)
```

    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    


```python
def eqdi(pt1,pt2,xi=1):
    result = np.linalg.norm(pt1-pt2,axis=xi)
    return result
```


```python
err = eqdi(c,ctold,None)
print(err)
```

    12.53714481052205
    


```python
loop = 0
```


```python
import copy
while err!=0:
    loop = loop+1
    for i in range(len(a)):
        distance = eqdi(a[i],c)
        cluster = np.argmin(distance)
        op[i] = cluster
    ctold = copy.deepcopy(c)
    print("old centroid")
    print(ctold)
    for l in range(k):
        pts = [a[j] for j in range(len(a)) if op[j] == l]
        c[l] = np.mean(pts, axis=0)
    print("New Centroids after ",loop,"Iteration \n", c)
    err = eqdi(c, ctold, None)
    print("Error - ",err)
    print("Clusters")
    print(op)
    
```

    old centroid
    [[6.2 3.2]
     [6.6 3.7]
     [6.5 3. ]]
    New Centroids after  1 Iteration 
     [[5.17142857 3.17142857]
     [5.5        4.2       ]
     [6.45       2.95      ]]
    Error -  1.588639515498743
    Clusters
    [0. 0. 2. 0. 1. 0. 0. 2. 0. 0.]
    old centroid
    [[5.17142857 3.17142857]
     [5.5        4.2       ]
     [6.45       2.95      ]]
    New Centroids after  2 Iteration 
     [[4.8   3.05 ]
     [5.3   4.   ]
     [6.2   3.025]]
    Error -  0.548478879841925
    Clusters
    [2. 0. 2. 0. 1. 0. 0. 2. 1. 2.]
    old centroid
    [[4.8   3.05 ]
     [5.3   4.   ]
     [6.2   3.025]]
    New Centroids after  3 Iteration 
     [[4.8   3.05 ]
     [5.3   4.   ]
     [6.2   3.025]]
    Error -  0.0
    Clusters
    [2. 0. 2. 0. 1. 0. 0. 2. 1. 2.]
    


```python
import matplotlib.pyplot as mathpl
```


```python
pts = [a[j] for j in range(len(a)) if op[j] == 0]
mathpl.scatter(*zip(*pts),c='red')
```




    <matplotlib.collections.PathCollection at 0x1d52469deb0>




    
![png](output_9_1.png)
    



```python
pts = [a[j] for j in range(len(a)) if op[j] == 1]
mathpl.scatter(*zip(*pts),c='green')
```




    <matplotlib.collections.PathCollection at 0x1d5267b7af0>




    
![png](output_10_1.png)
    



```python
pts = [a[j] for j in range(len(a)) if op[j] == 2]
mathpl.scatter(*zip(*pts),c='blue')
```




    <matplotlib.collections.PathCollection at 0x1d52683a640>




    
![png](output_11_1.png)
    



```python
for j in range(len(a)):
    if op[j] == 0:
        mathpl.scatter(a[j,0],a[j,1],c="red")
    elif op[j] == 1:
        mathpl.scatter(a[j,0],a[j,1],c="green")
    elif op[j] == 2:
        mathpl.scatter(a[j,0],a[j,1],c="blue")
        
```


    
![png](output_12_0.png)
    



```python

```
