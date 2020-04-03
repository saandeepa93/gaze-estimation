import numpy as np

a = np.array([9, 9, 5, 5, 5, 7, 7, 8, 9 , 9, 9 , 9 ])
temp = (np.unique(a, return_counts = True))
print(temp[0][np.argmax(temp[1])])