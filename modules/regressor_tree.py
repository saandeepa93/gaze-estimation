import numpy as np
import sys

class __Node__:
  def __init__(self, val=0, left=None, right=None, \
    num_samples=0, class_freq=None):
    self.val = val
    self.left = None
    self.right = None
    self.num_samples = num_samples
    self.class_freq = class_freq
    self.score = np.inf

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
  def __init__(self, depth, min_samples = 5):
    self.depth = depth
    self.tree_depth = 0
    self.min_samples = min_samples

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



  def __best_split__(self, X, y):

    """Best split
        y - np.array()
        self.targets - list

    Returns:
        @best_feature -- The feature on which to split
        @best_threshold -- The apt threshold on which to condition at each level
    """
    n = y.shape[0]
    best_feature, best_threshold = None, None
    score = np.inf
    if n > 1:
      for i in range(self.num_features):
        t = X[:,i]
        for j in range(n):
          lhs = t <= t[j]
          rhs = t > t[j]
          if np.sum(lhs) < self.min_samples or np.sum(rhs) < self.min_samples:
            continue
          new_score = np.std(y[lhs]) * np.sum(lhs) + np.std(y[rhs]) * np.sum(rhs)
          if new_score < score:
            score = new_score
            best_feature = i
            best_threshold = t[j]

    return best_feature, best_threshold, score

  def __build_tree__(self, X, y, depth):
    z = y.shape[0]
    n = __Node__(num_samples=z)
    n.class_freq = np.unique(y, return_counts = True)
    n.val = np.average(y)
    if depth < self.depth:
      if depth > self.tree_depth:
        self.tree_depth = depth
      n.feature, n.threshold, n.score = self.__best_split__(X, y)
      if n.score != np.inf:
        i = X[:, n.feature] < n.threshold # Difference between intensities of 2 pixels
        X_left, y_left = X[i, :], y[i]
        X_right, y_right = X[~i,:], y[~i]
        if X_left.shape[0] != 0:
          n.left = self.__build_tree__(X_left, y_left, depth+1)
        if X_right.shape[0] != 0:
          n.right = self.__build_tree__(X_right, y_right, depth+1)
    return n

  def fit(self, X, y):
    return
    self.targets = np.unique(y)
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
