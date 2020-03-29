import numpy as np
import cv2
from numpy import cos, sin, pi
import sys

def show_image(img):
  cv2.imshow("image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def generate_lms():
  source = np.zeros((320,320,3))
  num_squares = 16
  lm = np.zeros((16,2))
  inc = int(source.shape[0]/num_squares * 4)
  patch_expert = np.zeros((num_squares, 3))
  counter = 0
  for i in range(0, source.shape[0], inc):
    for j in range(0, source.shape[1], inc):
      source[i:i+inc,j:j+inc,0] = i/255.0
      source[i:i+inc,j:j+inc,1] = j/255.0
      source[i:i+inc,j:j+inc,2] = ((i+j)%255)/255.0
      cv2.circle(source, (j+(inc//2), i+(inc//2)), 5, (255,255,255), 2)
      lm[counter,0] = j+(inc//2)
      lm[counter,1] = i+(inc//2)
      patch_expert[counter] = np.array([i/255.0, j/255.0, ((i+j)%255)/255.0])
      counter+=1
      # show_image(source)


  return lm - np.average(lm,axis=0), patch_expert, source

def patch(triplet, patch_expert):
  return np.argmin((np.sqrt(np.sum(np.square(patch_expert - triplet), axis=1))))

def get_sim_matrix(img, patch_expert, lm):
  lm_1 = np.copy(lm)
  counter = 0
  for lms in lm.astype(np.int):
    triplet = img[lms[1]][lms[0]]
    lm_1[counter] = lm[patch(triplet, patch_expert)]
    counter+=1
  # lm_inv = (np.matmul(np.linalg.inv(np.matmul(lm.T,lm)),lm.T))
  lm_inv = np.matmul(np.linalg.inv(np.matmul(lm.T, lm)),lm.T)
  print(np.matmul(lm_inv,lm_1))
  return np.matmul(lm_inv,lm_1)
  return np.array([
    [cos(90 * pi/180), -sin(90 * pi/180), 0],
    [sin(90 * pi/180), cos(90 * pi/180), 0],
    [0, 0, 1]
  ])

def pred_lms(img, patch_expert, lm):
  lm = np.hstack((lm, np.ones((16,1))))
  lm =  (np.matmul(lm, get_sim_matrix(img, patch_expert, lm)))[:,:2]
  return lm


lm, patch_expert, source = generate_lms()
source_rot90 = cv2.rotate(source, cv2.ROTATE_180)

# show_image(source)
# show_image(source_rot90)
centroid = np.array([160,160])
pred_lm = pred_lms(source_rot90, patch_expert, lm + centroid)
print("\n")
for l, lp in zip(lm, pred_lm):
  print(f"orig:{l}, pred:{lp - centroid}")