Python 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 19:29:22) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
>>> data=np.array([[1,2],[3,4],[5,6]])
>>> data
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> data[0,1]
2
>>> data[1:3]
array([[3, 4],
       [5, 6]])
>>> data[0:2,0]
array([1, 3])
>>> data.max()
6
>>> data.sum()
21
>>> data*2.0
array([[ 2.,  4.],
       [ 6.,  8.],
       [10., 12.]])
>>> 

>>> data.max(axis=0)
array([5, 6])
>>> data.max(axis=1)
array([2, 4, 6])
>>> 
