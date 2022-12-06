```python
import math
arr = [115.3,195.5,120.5,110.2,90.4,105.6,110.9,116.3,122.3,125.4]
sum = 0
for i in arr :
    sum = sum + i
avg = sum/10
arr.sort()
mid = len(arr) // 2
res = (arr[mid] + arr[mid+1]) / 2
print(avg)
print(res)
count = 1
m = 0
for i in range(0,10):
    for j in range(i+1,10):
        if arr[i] == arr[j]:
            count = count + 1
    if count>m:
        m=count
        mod=arr[i]
        
if m==1:
    print("No mode")
else:
    print("Mode is ",mod)
var = 0
for i in arr:
    diff=i-avg
    var = var+(diff**2)
print("Variance is ",var)
std=math.sqrt(var)
print("Standard Deviation is ",std)
a = arr[9]-arr[0]
print("Min Max normalization")
for i in arr:
    diff = (i-arr[0])/a
    print(diff)
print("Standardization")
for i in arr:
    stdev = (i-avg)/std
    print(stdev)
    

```

    121.24000000000001
    118.4
    No mode
    Variance is  7017.724
    Standard Deviation is  83.7718568494217
    Min Max normalization
    0.0
    0.14462416745956222
    0.18839200761179828
    0.1950523311132255
    0.23691722169362506
    0.24643196955280677
    0.2863939105613701
    0.3035204567078972
    0.3330161750713606
    1.0
    Standardization
    -0.3681427290722982
    -0.1866975448343304
    -0.1317865022359979
    -0.12343047401451247
    -0.07090686805089026
    -0.05896968487733974
    -0.008833515548427498
    0.012653414163963414
    0.04965868200197015
    0.8864552224678619



```python

```
