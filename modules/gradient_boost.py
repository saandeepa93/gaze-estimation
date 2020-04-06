import numpy as np

class GradientBoost:
  def __init__(self, c, models = []):
    self.c = c
    self.models = models


  def update(self, eta, r):
    self.models.append((eta, r))


  def save_model(self):
    #TODO figure this out
    pass


  def predict(self, I, S):
    result = self.c
    for (eta, model) in self.models:
      result += (eta * model.predict((I, S))[0])
    return result