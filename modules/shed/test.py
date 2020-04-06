import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import sys
# import modules.transformation as tran
import cv2
import skimage.transform as tf


lm = np.array([
  [2, 2],
  [4, 4],
  [6, 6],
  [8, 8],
  [2, 4],
  [3, 6]
])

theta = 45 * np.pi/180

s = 2
r = np.array([
  [s * cos(theta), s * -sin(theta), 5],
  [s * sin(theta),s * cos(theta), 1],
  [0, 0, 1],
])
print(r)
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
], np.float32)

A = np.array([
 [ -3,   5],
 [ -7,   9],
 [-11,  13],
 [-15,  17]
], np.float32)

A = lm2.T.astype(np.float32)

# A = np.hstack((A, np.ones((A.shape[0], 1)))).astype(float)
# B = np.hstack((B, np.ones((B.shape[0], 1)))).astype(float)
# P = np.matmul(np.matmul(np.linalg.inv(np.matmul(B.T, B)), B.T), A).round(1).T


P = tf.estimate_transform('similarity' ,B, A)
print(P.params)

sys.exit(0)
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





