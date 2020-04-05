import numpy as np
import math
import glob
import pandas as pd
import sys
import random
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor as DecisionTreeRegressor2

from modules.preprocess import preprocessing
from modules.triplet_init import init, load_train
from modules.gradient_descent import train_model
from modules.onemillisecond import OneMS
from modules.transformation import calc_transformation
from modules.regressor_tree import RegressorTree
from modules.shed.regressor_net import DecisionTreeRegressor


def step_1():
  # pre-processing
  n = len(glob.glob1('./input/raw/train/', '*.png'))
  preprocessing( './input/raw/train/', './input/train/')


def step_2():
  # raw to processed data
  filepath = './input/train/'
  triplet = np.array(init(5, 447, filepath))
  np.save('./input/train/triplets.npy', triplet)


def denormalize(lms, mean_frontal):
    lms[:,0] *= ((mean_frontal[1][0] - mean_frontal[0][0]))
    lms[:,1] *= ((mean_frontal[1][1] - mean_frontal[0][1]))
    lms += mean_frontal[0]
    return lms.astype(np.int)


def transformation():
  #transformation.py
  triplet = np.load('./input/train/triplets.npy', allow_pickle = True)
  # img_dlib = denormalize(triplet[0][0][1], mean_frontal)
  # d = decision(triplet[0][0][0], img_dlib, (1, np.array([50, 127]),\
  #   np.array([25, 72])))


def grad_descent():
  X_train = load_train('./input/train/triplets.npy', 446)
  train_model(X_train)


def gradient_boost():
  #onemillisecond.py
  t = 1
  r = None

  mean_landmarks = np.load('./input/train/landmark_mean.npy')
  mean_frontal = np.load('./input/train/frontal/frontal_mean.npy')
  mean_landmarks = denormalize(mean_landmarks, mean_frontal)

  X_train = load_train('./input/train/triplets.npy', 446)
  oneMS = OneMS(mean_landmarks, mean_frontal, 0.3, 10)
  oneMS.fit(X_train)

def main():
  # step_1()
  # step_2()
  # transformation()
  gradient_boost()


if __name__ == '__main__':
  main()
