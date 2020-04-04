import numpy as np

def zipsort(a, b):
  ab = np.hstack((a, b[:,np.newaxis]))
  ab = ab[ab[:,-1].argsort()]
  return ab[:, :-1], ab[:,-1]



a = np.array([[1, 2, 3, 4, 5], [4, 3, 5, 2, 6], [2, 5, 2, 1, 7]])
b = np.array([1, 5, 1])
print(zipsort(a, b))