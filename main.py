import numpy as np
from numpy import sin as sin, cos as cos, genfromtxt
import cv2
import dlib
import pickle as rick
import glob
import os
import sys
import pandas as pd

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

def show_image(img):
  cv2.imshow("image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def get_features(img):
  dets = detector(img, 1)
  return [(d, predictor(img, d)) for _,d in enumerate(dets)]

def get_landmarks(frame):
  dd = get_features(frame)
  labels = np.zeros((7,2))
  for d,shape in dd:
    indices_3 = [2, 14, 8, 28, 33]
    counter = 2
    labels[0] = np.array([d.left(), d.right()])
    labels[1] = np.array([d.top(), d.bottom()])
    for i in indices_3:
        labels[counter] = np.array([shape.part(i).x,shape.part(i).y])
        cv2.circle(frame, (shape.part(i).x,shape.part(i).y), 3, (0,0,255), 2)
        counter+=1
        cv2.rectangle(frame, (d.right(), d.bottom()), (d.left(), d.top()), (255,0,0), 2)
  show_image(frame)
  return dd, labels.reshape(labels.shape[0] * labels.shape[1])

def create_patch(filepath, psize):
  labels_all = genfromtxt('./output/labels_train.csv', delimiter=',')
  tot_train = 50 #len(glob.glob1(filepath,"*.png"))
  patch_train = np.zeros((tot_train, 2*psize + 1, 2*psize, 3))
  counter = 0
  for fname in sorted(glob.glob(os.path.join(filepath,"*.png"))):
    patch_train[counter] = np.array([fname, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20])
    # img = cv2.imread(fname)
    # for i in range(4, 14, 2):
    #   x,y = labels_all[counter,i].astype(np.int), labels_all[counter,i+1].astype(np.int)
    #   xmin, xmax, ymin, ymax = x-psize, x+psize, y-psize, y+psize
    #   cv2.rectangle(img, (xmax, ymax),(xmin, ymin), (255,0,0), 2)
    #   show_image(img)
    #   patch_train[counter] = img[y-psize:y+psize, x-psize:x+psize,:]

    if counter==tot_train:
      break

  print(patch_train.shape)

def save_labels(flag=True):
  landmarks = np.zeros((38,68,2))
  if flag:
    cnt = 0
    filepath = "../input/neutral/train"
    num = len(glob.glob1(filepath,"*.png"))
    labels_all = np.zeros((num, 14))
    print("saving labels")
    for fname in sorted(glob.glob(os.path.join(filepath,"*.png"))):
      frame = cv2.imread(fname)
      dd, labels = get_landmarks(frame)
      labels_all[cnt] = labels
      print(fname.split('/')[-1], ":", labels_all[cnt])
      cnt+=1
      if cnt == 50:
        break
  r, c= labels_all.shape
  labels_all_mean = np.average(labels_all[:,4:], axis = 0)
  np.savetxt("./output/labels_train.csv", labels_all, delimiter=",")
  return labels_all


def main():
  labels_all = save_labels(1)

  num_subjects = len(os.listdir('./input/neutral/train/'))
  td = []

  cols = ['filename']
  cols.extend([f'm{i}' for i in range(14)])
  tdd = pd.DataFrame(columns=cols)

  for fname in sorted(glob.glob(os.path.join('./input/neutral/train',"*.png"))):
    a = get_features(cv2.imread(fname))
    box = a[0][0]
    temp = [fname, box.top(), box.left(), box.right(), box.bottom()]
    landmarks = []
    for i in [2, 14, 8, 28, 33]:
      landmarks.append(a[0][1].part(i).x)
      landmarks.append(a[0][1].part(i).y)
    temp.extend(landmarks)
    print(temp)
    # print(fname, box)
    # x_min, y_min, x_max, y_max =
    break


  with open('./tx.txt', 'w') as f:
    rick.dumps(td, f)

  #     print(fname.split('/')[-1] + " done")
  #     counter = 0
  #     img = cv2.imread(fname)
  #     for i in range(4, 14, 2):
  #       x,y = labels_all[counter,i].astype(np.int), labels_all[counter,i+1].astype(np.int)
  #       if fname.split('/')[-1] == 'Sub28_vid_1_frame30.png':
  #         print(x, y)
  #       cv2.circle(img, (x,y), 3, (0,0,255), 2)
  #       show_image(img)
  #       counter+=1

  # create_patch("./input/neutral/train", 15)

if __name__ == '__main__':
  main()