import numpy as np

class GradientBoost:
  def __init__(self, c, models = []):
    self.c = c
    self.models = models


  def update(self, eta, r):
    self.models.append((eta, r))


  def print(self):
    print(len(self.models))

  def predict(self, I, S):
    result = np.copy(self.c)
    for (eta, model) in self.models:
      result += (eta * model.predict(np.array([(I, S)]))[0])
    return result