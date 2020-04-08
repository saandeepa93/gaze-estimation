import numpy as np
import joblib

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
  oneMS = OneMS(mean_landmarks, eta = 0.3, T = 1, K = 2, d = 5, t_count = 20)
  oneMS.fit(X_train)
  oneMS.print()
  joblib.dump(oneMS, './models/best_model.npy')


def main():
  # step_1()
  # step_2()
  one_millisecond()
  # m = joblib.load('./models/best_model.npy')
  # m.print()


if __name__ == '__main__':
  main()
