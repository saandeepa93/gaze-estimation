import numpy as np
import glob
from modules.preprocess import preprocessing
from modules.triplet_init import init, load_train
from modules.train import train_model
from modules.onemillisecond import OneMS
from modules.transformation import calc_transformation
import pandas as pd


def denormalize(lms, mean_frontal):
  lms[:,0] *= ((mean_frontal[1][0] - mean_frontal[0][0]))
  lms[:,1] *= ((mean_frontal[1][1] - mean_frontal[0][1]))
  lms += mean_frontal[0]
  return lms.astype(np.int)

mean_landmarks = np.load('./input/train/landmark_mean.npy')
mean_frontal = np.load('./input/train/frontal/frontal_mean.npy')
mean_landmarks = denormalize(mean_landmarks, mean_frontal)


def get_primes(pts, mean_landmarks, mean_frontal, S):
  closest_lm = np.zeros((pts.shape[0], 2))
  dX = np.zeros((pts.shape[0], 2))
  cur_lm = np.zeros((pts.shape[0], 2))
  primes = np.zeros((pts.shape[0], 2))
  pts[:,0] += mean_frontal[0][0]
  pts[:,1] += mean_frontal[0][1]
  cnt = 0

  s, R, t = calc_transformation(mean_landmarks, S)
  R_tmp = np.copy(R)
  R_tmp[0,:] = R[0,:]/s[0]
  R_tmp[1,:] = R[1,:]/s[1]
  #TODO verify calculations
  for i in pts:
    temp = np.sum(np.square(mean_landmarks - i), axis = 1)
    ind = np.argmin(temp)
    dX[cnt,:] = mean_landmarks[ind,:] - S[ind,:]
    closest_lm[cnt,:] = mean_landmarks[ind, :]
    cur_lm[cnt,:] = S[ind, :]
    cnt+=1

  return (cur_lm.T + np.matmul(R_tmp.T, dX.T)).T


def decision(I, shpe, theta):
  """Given the input, the function returns a decision 0 or 1

  Arguments:
      I {np.array} -- Image
      shpe {np.array} -- current estimate shape for I
      theta {triplet} -- triplet of threshold and 2 locations: u,v
  """
  pts = np.array([
    theta[1],
    theta[2]
  ])

  primes = get_primes(pts, mean_landmarks, mean_frontal, shpe).astype(np.int)
  #TODO handle out of bound prime indeces
  dist = I[primes[0][0]][primes[0][1]] - I[primes[1][0]][primes[1][1]]
  return int(np.sqrt(np.sum(np.square(dist))) > theta[0])


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
  # t = 1
  # r = None
  # X_train = load_train('./input/train/triplets.npy', 446)
  # oneMS = OneMS(0.3, 10)
  # oneMS.fit(X_train)


  #transformation.py
  # triplet = np.load('./input/train/triplets.npy', allow_pickle = True)
  # img_dlib = denormalize(triplet[0][0][1], mean_frontal)
  # d = decision(triplet[0][0][0], img_dlib, (1, np.array([50, 127]),\
  #   np.array([25, 72])))

  X = pd.read_csv('./input/POC_data/train.csv')
  y = np.array(X['Survived'])
  X = X.drop(['Survived', 'Cabin'], axis = 1)


  print(X)



if __name__ == '__main__':
  main()
