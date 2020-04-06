import numpy as np

from modules.onemillisecond import OneMS
from modules.preprocess import preprocessing
from modules.triplet_init import init, load_train


def step_1():
  # pre-processing
  preprocessing( './input/raw/train/', './input/train/')


def step_2():
  # raw to processed data
  filepath = './input/train/'
  triplet = np.array(init(5, 447, filepath))
  print(triplet.shape)
  np.save('./input/train/triplet/triplets.npy', triplet)


def one_millisecond():
  mean_landmarks = np.load('./input/train/frontal/landmark_mean.npy')
  X_train = load_train('./input/train//triplet/triplets.npy', 446)
  # oneMS = OneMS(mean_landmarks, eta = 0.3, T = 10, K = 500, d = 5, t_count = 500)
  oneMS = OneMS(mean_landmarks, eta = 0.3, T = 2, K = 5, d = 5, t_count = 5)
  oneMS.fit(X_train)


def main():
  # step_1()
  # step_2()
  one_millisecond()


if __name__ == '__main__':
  main()
