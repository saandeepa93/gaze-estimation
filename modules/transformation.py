import numpy as np
import skimage.transform as tf

def calc_transformation(mean_landmarks, img_dlib):
  rot = np.zeros((2, 2))
  trans = np.zeros((2, 1))
  scale = np.zeros((2, 1))


  P = tf.estimate_transform('similarity', img_dlib, mean_landmarks).params
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