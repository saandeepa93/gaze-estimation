import numpy as np
import pickle as rick
import joblib
from .regressor_tree import RegressorTree
from .gradient_boost import GradientBoost

class OneMS:
  def __init__(self, mean_landmarks, eta = 0.4, T = 10, K = 500, d = 5, \
    t_count = 500):
    self.mean_landmarks = mean_landmarks
    self.eta = eta
    self.T = T
    self.K = K
    self.d = d
    self.t_count = t_count


  def get_dS(self):
    return np.average(self.X_train[:,0,2], axis = 0)


  def fit(self, X_train):
    """
    This method trains an ensemble of models

    ### Arguments
      * `X_train` (`{np.array}`) Every element in an array of triplets
        * The first value in the triplet is an `m*n*3` image
        * The second value is an array of (68,2) landmark locations
        * The third value is an array of (68,2) SSD
    """

    self.X_train = X_train
    r = None
    for i in range(self.T):
      r = self.__cascade__(r)
      for ind in range(self.X_train.shape[0]):
        self.X_train[ind, 0, 1] += r.predict(self.X_train[ind, 0, 0],\
          self.X_train[ind, 0, 1])
        self.X_train[ind, 0, 2] = self.X_train[ind, 0, 4] -\
          self.X_train[ind, 0, 1]

    self.final_model = r

  def print(self):
    self.final_model.print()

  def predict(self, I):

    """
    This method returns m landmark locations for the image.

    ### Arguments
      * 'I' An image of size m*n*3.

    ### Returns
      * A (68*2) vector of landmarks
    """
    S = self.mean_landmarks
    return S + self.final_model.predict(I, S)


  def __cascade__(self, r = None):
    # print("cascade")
    N = 100#self.X_train.shape[0]
    f = GradientBoost(c = self.get_dS()) if r is None else r
    for k in range(self.K):
      print(k)
      r = np.zeros((N, 68, 2))
      X = []
      g = RegressorTree(self.d, self.t_count, self.mean_landmarks)
      for i in range(N):
        r[i,:,:] = self.X_train[i,0,2] - f.predict(self.X_train[i,0,0], \
          self.X_train[i,0,1])
        X.append((self.X_train[i,0,0], self.X_train[i,0,1], self.X_train[i,0,3]))
      g.fit(np.array(X), r)
      f.update(self.eta, g)
    return f
