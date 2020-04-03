import numpy as np


class F:
  def __init__(self, model):
    self.model = model
    # Here goes train_model for f0

  def update(self, new_model):
    self.model = new_model

  def predict(self, I, S):
    return self.model


class G:
  def __init__(self ):
    self.model = np.zeros((68,2))

  def update(self, r):
    pass

  def predict(self, I, S):
    return self.model


class OneMS:
  def __init__(self, eta = 0.4, t = 10):
    self.eta = eta
    self.t = t

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
    for i in range(self.t):
      r = self.__cascade__(r)

    #TODO: to be removed
    return r

  def predict(self, I):

    """
    This method returns m landmark locations for the image.

    ### Arguments
      * 'I' An image of size m*n*3.

    ### Returns
      * A (68*2) vector of landmarks
    """

    return self.model

  def __cascade__(self, r = None):
    N = self.X_train.shape[0]
    K = 10
    eta = 0.3
    f, g = F(self.get_dS() if r is None else r), G()
    for k in range(K):
      r = np.zeros((N, 68, 2))
      for i in range(N):
        r[i,:,:] = self.X_train[i,0,2] - f.predict(self.X_train[i,0,0], \
          self.X_train[i,0,1])
      g.update(r)
      f.update(f.model + eta * g.model)
    return f.model
