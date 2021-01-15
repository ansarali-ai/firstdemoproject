#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[11]:


np.zeros(10)


# 3. Create a vector with values ranging from 10 to 49

# In[13]:


arr1=np.arange(10,49)
arr1


# 4. Find the shape of previous array in question 3

# In[14]:


arr1.shape


# 5. Print the type of the previous array in question 3

# In[15]:


print(arr1.dtype)


# 6. Print the numpy version and the configuration
# 

# In[19]:


print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[23]:


print(arr1.ndim)


# 8. Create a boolean array with all the True values

# In[ ]:





# 9. Create a two dimensional array
# 
# 
# 

# In[27]:


np.arange(25).reshape(5,5)


# 10. Create a three dimensional array
# 
# 

# In[25]:


np.arange(64).reshape(4,4,4)


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[32]:


x=np.arange(5)
print(x)
np.flip(x)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[36]:


arr2=np.zeros(10)
arr2
arr2[5]=1
arr2


# 13. Create a 3x3 identity matrix

# In[38]:


np.identity(3)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[45]:


arr=np.array([1,2,3,4,5])
arr.astype(np.float)


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[46]:


arr1=np.array([[1.,2.,3.],[4.,5.,6.]])
arr2=np.array([[0.,4.,1.],[7.,2.,12.]])
arr1*arr2


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[47]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])
np.maximum(arr1,arr2)


# 17. Extract all odd numbers from arr with values(0-9)

# In[49]:


arr=np.arange(10)
arr[arr%2==1]


# 18. Replace all odd numbers to -1 from previous array

# In[51]:


arr[arr%2==1]=-1
arr


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[56]:


arr=np.arange(10)
arr[arr[5:9]]=12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[60]:


d2=np.ones((4,4))
d2[1:-1,1:-1]=0
d2


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[63]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
arr2d[1,1]=12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[82]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d


# In[83]:


arr3d[0,0:]=64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[97]:


arr3=np.arange(10).reshape(5,2)
print(arr3)
arr3[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[96]:


arr4=np.arange((10)).reshape(5,2)
arr4


# In[98]:


arr4[1,1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[101]:


arr5=np.arange(10).reshape(2,5)
print(arr5)
arr5[0:2,2:3]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[109]:


arr6=np.random.randn(10,10)
print(arr6)
print(f"minimum value is{arr6.min()}")
print(f"maximum value is{arr6.max()}")


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[111]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[112]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a==b)


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[114]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
print(data)
data[names!='Will']


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[125]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
print(data)
data[(names !='Will') & (names !='Joe')]


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[136]:


np.arange(1,16).reshape(5,3).astype(np.float)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[138]:


x=np.arange(1,17).reshape(2,2,4)
x.astype(float)


# 33. Swap axes of the array you created in Question 32

# In[146]:


x.T


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[160]:


a=np.arange(10)
b=np.sqrt(a)
b[b<0.5]=0
b


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[165]:


x=np.random.randn(12)
y=np.random.randn(12)
print(x)
print(y)
np.maximum(x,y)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[166]:


np.unique(names)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[167]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
np.setdiff1d(a,b)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[173]:


sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = numpy.array([[10,10,10]])
sampleArray[:,1]=newColum[:,0]


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[174]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x.dot(y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[178]:


d=np.random.randn(20)
d.cumsum()


# In[ ]:




