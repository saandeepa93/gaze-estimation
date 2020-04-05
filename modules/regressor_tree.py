import numpy as np
import random
import sys

from .transformation import calc_transformation

class __Node__:
  def __init__(self, val=0, left=None, right=None, num_samples=0):
    self.val = val
    self.left = None
    self.right = None
    self.num_samples = num_samples
    self.score = 0

  def print(self):
    print("self: ",self)
    print("class: ", self.val)
    print("threshold: ", self.threshold)
    print("num_samples: ", self.num_samples)
    print("feature: ", self.feature)
    print("left: ", self.left)
    print("right: ", self.right)
    print("score: ", self.score)


class RegressorTree:
  def __init__(self, depth, mean_landmarks, mean_frontal, min_samples = 5):
    self.depth = depth
    self.tree_depth = 0
    self.min_samples = min_samples
    self.mean_landmarks = mean_landmarks
    self.mean_frontal = mean_frontal

  def __zipsort__(self, a, b):
    ab = np.hstack((a[:,np.newaxis], b[:,np.newaxis]))
    ab = ab[ab[:,0].argsort()]
    return ab[:, 0], ab[:,-1].astype(np.int)

  def print_tree(self, root):
    if root.left is not None:
      self.print_tree(root.left)
    print("---------------------")
    root.print()
    print("---------------------")
    if root.right is not None:
      self.print_tree(root.right)



  def __get_primes__(self, pts, mean_landmarks, mean_frontal, S):
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



  def __decision__(self, I, shpe, theta):
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

    primes = self.__get_primes__(pts, self.mean_landmarks, self.mean_frontal, shpe).astype(np.int)
    #TODO handle out of bound prime indeces
    dist = I[primes[0][0]][primes[0][1]] - I[primes[1][0]][primes[1][1]]
    return int(np.sqrt(np.sum(np.square(dist))) > theta[0])





  def __best_split__(self, X, y):

    """Best split
        y - np.array()
        self.targets - list

    Returns:
        @best_feature -- The feature on which to split
        @best_threshold -- The apt threshold on which to condition at each level
    """
    n, c, _ = y.shape
    best_feature, best_threshold = None, None
    score = 0
    if n > 1:
      thetas = [(random.sample(range(1,100), 1)[0], b[0:2], b[2:]) for b in np.random.rand(10, 4)]
      for i in thetas:
        t = np.arange(n)
        mu = np.sum(y, axis = 0)
        lhs = []
        rhs = []
        for observation in range(n):
          if self.__decision__(X[observation,:][0], X[observation,:][1], i) == 0:
            lhs.append(observation)
          else:
            rhs.append(observation)
        num_lhs = len(lhs)
        num_rhs = len(rhs)
        if num_lhs < self.min_samples or num_rhs < self.min_samples:
          continue
        mu_l = np.sum(y[lhs], axis = 0)/num_lhs
        mu_r = (mu - num_lhs * mu_l)/num_rhs
        error_l = num_lhs * np.sum(np.sqrt(np.sum(np.square(mu_l), axis = 1)))
        error_r = num_rhs * np.sum(np.sqrt(np.sum(np.square(mu_r), axis = 1)))
        new_score = np.sum(error_l + error_r)
        if new_score > score:
          score = new_score
          best_feature = i
          best_threshold = (X[lhs], y[lhs], X[rhs], y[rhs])
    return best_feature, best_threshold, score

  def __build_tree__(self, X, y, depth):
    z = y.shape[0]
    n = __Node__(num_samples=z)
    n.val = np.average(y, axis = 0)
    if depth < self.depth:
      n.feature, n.threshold, n.score = self.__best_split__(X, y)
      if n.threshold is not None:
        if n.threshold[0].shape[0] != 0:
          n.left = self.__build_tree__(n.threshold[0], n.threshold[1], depth+1)
        if n.threshold[2].shape[0] != 0:
          n.right = self.__build_tree__(n.threshold[2], n.threshold[3], depth+1)
    return n

  def fit(self, X, y):
    self.num_features = X.shape[1]
    self.root = self.__build_tree__(X, y, 0)

  def __predict_per_item__(self, item):
    start = self.root
    while start is not None and start.left is not None:
      if item[start.feature] < start.threshold:
        start = start.left
      else:
        start = start.right
    return start.val

  def predict(self, X):
    return
    return [self.__predict_per_item__(i) for i in X]