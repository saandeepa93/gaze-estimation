import numpy as np


class Node:
  def __init__(self):
    self.left = None
    self.right = None
    self.val = 0

lst = [0, 2, 4, 5, 7, 9, 11, 13, 15, 23, 34, 56, 22]
bin_lst = []

def get_tree(i):
  if i >= len(lst):
    return None
  node = Node()
  node.val = lst[i]
  node.left = get_tree(2*i+1)
  node.right = get_tree(2*i+2)
  return node

def get_list(root):
  ls2 = []
  res = []
  ls2.append(root)
  res.append(root.val)
  root.is_visited = True
  while len(ls2) != 0:
    v = ls2.pop(0)
    if (v.left is not None) and ((~hasattr(v.left, 'is_visited') or (v.left.is_visited == False))):
      v.left.is_visited = True
      res.append(v.left.val)
      ls2.append(v.left)
    if (v.right is not None) and ((~hasattr(v.right, 'is_visited') or (v.right.is_visited == False))):
      v.right.is_visited = True
      res.append(v.right.val)
      ls2.append(v.right)
  return res

def print_tree(root):
  if root.left is not None:
    print_tree(root.left)
  print("---------------------")
  print(root.val)
  print("---------------------")
  if root.right is not None:
    print_tree(root.right)

# r = get_list(get_tree(0))
# np.save('./models/test.npy', r)

lst = np.load('./models/test.npy',allow_pickle = True)
print(get_list(get_tree(0)))
