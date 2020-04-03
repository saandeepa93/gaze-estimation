import numpy as np
import cv2
import sys
import cv2

def cost_func(dS, gamma):
  return np.sum(np.sqrt(np.sum(np.square(dS - gamma), axis = 1))), \
          dS - gamma

def load_img(val, val2, img):
  lm_mean = np.load('./input/train/frontal/frontal_mean.npy')
  xmin, ymin, xmax, ymax = lm_mean[0][0], lm_mean[0][1], lm_mean[1][0], lm_mean[1][1]
  gamma = (val * np.array([xmax - xmin, ymax - ymin]) + np.array([xmin, ymin])).astype(np.int)
  dS = (val2 * np.array([xmax - xmin, ymax - ymin]) + np.array([xmin, ymin])).astype(np.int)

def train_model(X_train):
  eta = np.array([0.1, 0.1])
  dS = np.average(X_train[:,0,2], axis = 0)
  gamma_old = gamma_new = np.random.rand(68, 2)

  delta_gamma = np.ones((68, 2))
  while True:
    img = cv2.imread('./input/raw/train/frame1.png')
    load_img(gamma_old, gamma_old, img)

    cost, gradient = cost_func(dS, gamma_old)
    gamma_new = gamma_old + eta * gradient
    new_cost, new_gradient = cost_func(dS, gamma_new)
    if new_cost >= cost:
      break
    cost = new_cost
    gamma_old = gamma_new

  return gamma_old, dS
