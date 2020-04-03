import numpy as np
from random import randint
import glob
import os
import sys


def load_train(filepath, n):
  train_data = np.zeros((n))
  cnt = 0
  temp = np.load(filepath, allow_pickle=True)
  return temp


def init(R, n, filepath):
  S, I, triplet = [], [], []
  for fname in glob.glob(os.path.join(filepath, "*.npy")):
    S.append(np.load(fname, allow_pickle=True)[0][1])
    I.append(np.load(fname, allow_pickle=True)[0][0])

  for i in range(n):
    s = set(np.arange(1,n))
    pi_i = s.pop()
    for j in range(R):
      S_i = S[s.pop()]
      dS_i = S[pi_i] - S_i
      triplet.append([(I[pi_i], S_i, dS_i)])
  return triplet
