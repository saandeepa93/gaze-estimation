import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import modules.transformation as tran


lm = np.array([
  [2, 2],
  [4, 4],
  [6, 6],
  [8, 8],
  [2, 4],
  [3, 6]
])

theta = 90 * np.pi/180

s = 2
r = np.array([
  [s * cos(theta), s * -sin(theta), 1],
  [s * sin(theta),s * cos(theta), 1],
  [0, 0, 1],
])
lm = np.hstack((lm, np.ones((lm.shape[0], 1))))
lm2 = (np.matmul(r, lm.T)[:-1,:])

# plt.show()

"""
our method
"""

#Not a full rank matrix
B = np.array([
  [2, 2],
  [4, 4],
  [6, 6],
  [8, 8],
  [2, 4],
  [3, 6]
])

A = np.array([
 [ -3,   5],
 [ -7,   9],
 [-11,  13],
 [-15,  17],
 [ -7,   5],
 [-11,   7]
])

A = np.hstack((A, np.ones((A.shape[0], 1))))
B = np.hstack((B, np.ones((B.shape[0], 1))))

print(A.shape, B.shape)
P = np.matmul(np.matmul(np.linalg.inv(np.matmul(B.T, B)), B.T), A).round(1).T

#TODO separate scale for x and y.
theta = np.arctan(P[1][0]/P[0][0])
print(theta * 180/pi)
scale = P[1][0] / sin(theta)
print(scale)

rot = np.array([
  [cos(theta), -sin(theta)],
  [sin(theta), cos(theta)]
])

print(rot.round(1))




# A_sq = np.matmul(A, A.T)
# u, s, V = np.linalg.svd(A_sq)
# s = np.diag(s)
# A_sq_inv = np.matmul(np.matmul(V, np.linalg.inv(s)), u.T)
# A_inv = np.matmul(A.T, A_sq_inv)

#Notes#





