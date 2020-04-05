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
from modules.train import train_model
from modules.onemillisecond import OneMS
from modules.transformation import calc_transformation
from modules.regressor_tree import RegressorTree
from modules.shed.regressor_net import DecisionTreeRegressor

def denormalize(lms, mean_frontal):
  lms[:,0] *= ((mean_frontal[1][0] - mean_frontal[0][0]))
  lms[:,1] *= ((mean_frontal[1][1] - mean_frontal[0][1]))
  lms += mean_frontal[0]
  return lms.astype(np.int)


mean_landmarks = np.load('./input/train/landmark_mean.npy')
mean_frontal = np.load('./input/train/frontal/frontal_mean.npy')
mean_landmarks = denormalize(mean_landmarks, mean_frontal)


def main():
  # n = len(glob.glob1('./input/raw/train/', '*.png'))
  # preprocessing( './input/raw/train/', './input/train/')


  # filepath = './input/train/'
  # triplet = np.array(init(5, 447, filepath))
  # np.save('./input/train/triplets.npy', triplet)

  #train.py
  # X_train = load_train('./input/train/triplets.npy', 446)
  # train_model(X_train)

  #onemillisecond.py
  t = 1
  r = None
  X_train = load_train('./input/train/triplets.npy', 446)
  oneMS = OneMS(mean_landmarks, mean_frontal, 0.3, 10)
  oneMS.fit(X_train)


  #transformation.py
  # triplet = np.load('./input/train/triplets.npy', allow_pickle = True)
  # img_dlib = denormalize(triplet[0][0][1], mean_frontal)
  # d = decision(triplet[0][0][0], img_dlib, (1, np.array([50, 127]),\
  #   np.array([25, 72])))

  X = pd.read_csv('./input/POC_data/train.csv')
  y = np.array(X['Survived'])
  X = X.drop(['Survived', 'Cabin'], axis = 1)


def test():
  X, y = load_boston(return_X_y=True)
  s = 400
  arr_pd = pd.read_csv('./input/POC_data/bris.csv', header = None)
  arr_pd[4] = pd.factorize(arr_pd[4])[0]
  arr = arr_pd.to_numpy()
  # np.random.shuffle(arr)
  # X_train = arr[:,:-1]
  # y_train = arr[:,-1]
  arr = np.hstack((X,y[:,np.newaxis]))

  X_train = arr[:s, :-1]
  y_train = arr[:s, -1]
  X_test = arr[s:, :-1]
  y_test = arr[s:, -1]

  # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
  r = RegressorTree(10)
  r_net = DecisionTreeRegressor()
  r_net2 = DecisionTreeRegressor2()

  r.fit(X_train, y_train)
  r_net.fit(X_train, y_train)
  r_net2.fit(X_train, y_train)
  # r.print_tree(r.root)

  y_pred = r.predict(X_test)
  y_pred_net = r_net.predict(X_test)
  y_pred_net2 = r_net2.predict(X_test)

  y_mean = np.average(y_test)
  u = np.sum(np.square(y_test - y_pred))
  u_net = np.sum(np.square(y_test - y_pred_net))
  u_net2 = np.sum(np.square(y_test - y_pred_net2))
  v = np.sum(np.square(y_test - y_mean))
  print(1 - u/v)
  print(1 - u_net/v)
  print(1 - u_net2/v)
  # print(np.round(y_pred,1))
  # print(y_test)

  # print(((sum(r.predict(X_test) == y_test)/ 25) * 100).round(2))


if __name__ == '__main__':
  main()
  # test(

