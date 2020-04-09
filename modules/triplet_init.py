import numpy as np
import glob
import os
import random


def load_train(filepath, n):
  train_data = np.zeros((n))
  cnt = 0
  temp = np.load(filepath, allow_pickle=True)
  return temp



def init(R, n, filepath):
  S, I, F, triplet = [], [], [], []
  for fname in glob.glob(os.path.join(filepath, "*.npy")):
    S.append(np.load(fname, allow_pickle=True)[0][1])
    I.append(np.load(fname, allow_pickle=True)[0][0])
    F.append(np.load(fname, allow_pickle=True)[0][2])


  for i in range(n):
    s = list(np.arange(1,n))
    random.shuffle(s)
    pi_i = s.pop()
    for j in range(R):
      S_i = S[s.pop()]
      dS_i = S[pi_i] - S_i
      triplet.append([(I[pi_i], S_i, dS_i, F[pi_i], S[pi_i])])
  return triplet
