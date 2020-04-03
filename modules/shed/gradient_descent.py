import numpy as np
import matplotlib.pyplot as plt
import random


def cost_func(v):
  return np.sum(np.sqrt(np.sum(np.square(v), axis = 1)))


def find_minima(x):
  #small change
  del_x = 2 * np.pi/180
  eta = 0.1
  plt.plot(x, cost_func(x))

  count, epoch = 0, 5
  global_min = 0

  while True:
    count += 1

    x_new = x_rand = random.randrange(0, 500) * np.pi/180
    plt.scatter(x_rand, cost_func(x_rand), color = 'orange')

    old_cost = 9e9

    while True:
      gradient = cost_func(x_rand + del_x) - cost_func(x_rand)

      x_new = x_new - eta * gradient
      plt.scatter(x_new, cost_func(x_new), color = 'green')

      if cost_func(x_new) > old_cost:
        break

      old_cost = cost_func(x_new)
      x_rand = x_new

    if cost_func(x_new) < cost_func(global_min):
      global_min = x_new

    if count == epoch:
      break

    plt.scatter(global_min, cost_func(global_min), color='red')
  plt.show()

find_minima(2)
