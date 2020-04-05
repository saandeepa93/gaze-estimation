import numpy as np
import sys

def calc_transformation(mean_landmarks, img_dlib):
  rot = np.zeros((2, 2))
  trans = np.zeros((2, 1))
  scale = np.zeros((2, 1))

  mean_landmarks = np.hstack((mean_landmarks, \
    np.ones((mean_landmarks.shape[0], 1)))).astype(np.int)

  img_dlib = np.hstack((img_dlib, \
    np.ones((img_dlib.shape[0], 1)))).astype(np.int)

  #TODO check if SVD is working
  return np.array([1, 1]), np.array([[1, 0], [0, 1]]), np.array([10.8, 0.55])
  M_sq = np.matmul(img_dlib.T, img_dlib)
  # U, S, V_t = np.linalg.svd(M_sq)
  # S_inv = np.linalg.inv(np.diag(S))
  # sys.exit(0)
  # M_inv = np.matmul(np.matmul(V_t.T, S_inv), U.T)
  P = np.matmul(np.matmul(M_sq, img_dlib.T), mean_landmarks).round(1).T

  theta = np.arctan(P[1][0]/P[0][0])


  if P[1][0] !=0 and P[1][0]/np.sin(theta) != 0:
    scale[0, 0] = P[1][0]/np.sin(theta)
  else:
    scale[0, 0] = P[0][0]/np.cos(theta)

  if P[1][1] !=0 and P[1][1]/np.cos(theta) != 0:
    scale[1, 0] = P[1][1]/np.cos(theta)
  else:
    scale[1, 0] = -P[0][1]/np.sin(theta)

  trans = P[:-1, -1]
  rot = P[:-1,:-1]

  return scale, rot, trans