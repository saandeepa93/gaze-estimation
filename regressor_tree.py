import numpy as np

class Node:
  def __init__(self, clas, left, right, threshold, gini, num_samples,\
    class_freq, feature=0):
    self.clas = clas
    self.left = None
    self.right = None
    self.threshold = threshold
    self.gini = gini
    self.num_samples = num_samples
    self.class_freq = class_freq
    self.feature = feature


class RegressorTree:

  def __init__(self, depth):
    self.depth = depth

  def __best_split__(self, X, y):
    pass

  def __gini__(self, y):
    freq = np.unique(y, return_counts = True)[1]
    return 1 - np.sum(np.square(freq/y.shape[0]))

  def __build_tree__(self, X, y, depth):
    z = y.shape[0]
    n = Node(num_samples=z, clas= np.argmax())
    n.gini = self.__gini__(y)
    n.class_freq = np.unique(y, return_counts = True)
    n.clas = n.class_freq[0][np.argmax(n.class_freq[1])]
    while depth < self.depth:
      feature, threshold = self.__best_split__(X, y)
      n.feature = feature
      n.threshold = threshold
      i = X[:, feature] < threshold
      X_left, y_left = X[i, :], y[i]
      X_right, y_right = X[~i,:], y[~i]
      n.left = self.__build_tree__(X_left, y_left, depth+1)
      n.right = self.__build_tree__(X_right, y_right, depth+1)
    return n

  def fit(self, X, y):
    self.targets = np.unique(y)
    self.num_features = X.shape[1]
    self.root = self.__build_tree__(X, y, 0)

  def __predict_per_item__(self, item):
    start = self.root
    while start.left is not None:
      if item[start.feature] < start.threshold:
        start = start.left
      else:
        start = start.right
    return start.clas

  def predict(self, X):
    return [self.__predict_per_item__(i) for i in X]