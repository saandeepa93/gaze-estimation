import numpy as np
import sys

class Node:
  def __init__(self, clas=0, left=None, right=None, threshold=0, gini=0,\
    num_samples=0, class_freq=None, feature=0):
    self.clas = clas
    self.left = None
    self.right = None
    self.threshold = threshold
    self.gini = gini
    self.num_samples = num_samples
    self.class_freq = class_freq
    self.feature = feature

  def print(self):
    print("self: ",self)
    print("class: ", self.clas)
    print("threshold: ", self.threshold)
    print("gini index: ", self.gini)
    print("num_samples: ", self.num_samples)
    print("feature: ", self.feature)
    print("left: ", self.left)
    print("right: ", self.right)


class RegressorTree:
  def __init__(self, depth):
    self.depth = depth
    self.tree_depth = 0

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
    if n > 1:
      g = self.__gini__(y)
      class_freq = np.array([np.sum(y==k) for k in range(len(self.targets))])
      for i in range(self.num_features):
        t, preds = self.__zipsort__(X[:,i], y.astype(np.int))
        left_freq = np.zeros((len(self.targets)))
        right_freq = np.copy(class_freq)
        for j in range(1,n):

          left_freq[preds[j-1]] +=1
          right_freq[preds[j-1]] -= 1
          left_gini = 1 - np.sum(np.square(left_freq/j))
          right_gini = 1 - np.sum(np.square(right_freq/(n-j)))
          if t[j] == t[j-1]:
            continue

          new_g = ((i * left_gini) + (n-i) * right_gini)/n
          if new_g < g:
            g = new_g
            best_feature = i
            best_threshold = (t[j] + t[j-1])/2

    return best_feature, best_threshold

  def __gini__(self, y):
    freq = np.unique(y, return_counts = True)[1]
    return 1 - np.sum(np.square(freq/y.shape[0]))

  def __build_tree__(self, X, y, depth):
    z = y.shape[0]
    n = Node(num_samples=z)
    n.gini = self.__gini__(y)
    n.class_freq = np.unique(y, return_counts = True)
    n.clas = n.class_freq[0][np.argmax(n.class_freq[1])]
    if depth < self.depth:
      if depth > self.tree_depth:
        self.tree_depth = depth
      feature, threshold = self.__best_split__(X, y)
      if feature is not None:
        n.feature = feature
        n.threshold = threshold
        i = X[:, feature] < threshold
        X_left, y_left = X[i, :], y[i]
        X_right, y_right = X[~i,:], y[~i]
        if X_left.shape[0] != 0:
          n.left = self.__build_tree__(X_left, y_left, depth+1)
        if X_right.shape[0] != 0:
          n.right = self.__build_tree__(X_right, y_right, depth+1)
    return n

  def fit(self, X, y):
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
    return start.clas

  def predict(self, X):
    return [self.__predict_per_item__(i) for i in X]
