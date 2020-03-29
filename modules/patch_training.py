import numpy as np
import cv2
import dlib
import glob
import os
import sys
import pickle

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, accuracy_score



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")

landmarks = [2, 14, 8, 28, 33]

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
  # show_image(frame)
  return dd, labels.reshape(labels.shape[0] * labels.shape[1])

def create_patch(filepath, oPath, psize):
  print(oPath)
  labels_all = np.loadtxt(oPath, delimiter=",")
  tot_train = len(glob.glob1(filepath,"*.png"))
  patch_train = np.zeros((tot_train, 2*psize, 2*psize, 3))
  typ = filepath.split('/')[-1]
  counter = 0
  for fname in sorted(glob.glob(os.path.join(filepath,"*.png"))):
    sub = fname.split('/')[-1]
    img = cv2.imread(fname)
    lm = 0
    for i in range(4, 14, 2):
      x,y = labels_all[counter,i].astype(np.int), labels_all[counter,i+1].astype(np.int)
      xmin, xmax, ymin, ymax = x-psize, x+psize, y-psize, y+psize
      # cv2.rectangle(img, (xmax, ymax),(xmin, ymin), (255,0,0), 2)
      patch_train[counter] = img[y-psize:y+psize, x-psize:x+psize,:]
      # show_image(patch_train[counter])
      cv2.imwrite("../output/patches/"+typ+"/"+str(landmarks[lm])+"/"+sub, patch_train[counter])
      lm = (lm + 1) % len(landmarks)
    counter+=1
    if counter==tot_train:
      break

def save_labels(filepath):
  # filepath = "../input/neutral/train"
  cnt = 0
  typ = filepath.split('/')[-1]
  num = len(glob.glob1(filepath,"*.png"))
  labels_all = np.zeros((num, 14))
  for fname in sorted(glob.glob(os.path.join(filepath,"*.png"))):
    frame = cv2.imread(fname)
    dd, labels = get_landmarks(frame)
    labels_all[cnt] = labels
    cnt+=1
    if cnt == num:
      break
  r, c= labels_all.shape
  labels_all_mean = np.average(labels_all[:,4:], axis = 0)
  output_file = "../output/labels_"+typ+".csv"
  np.savetxt(output_file, labels_all, delimiter=',', fmt='%d')
  return output_file

def build_model():
  model = Sequential()

  model.add(Conv2D(5, (3,3), input_shape = (30,30,3)))
  model.add(Activation('tanh'))

  model.add(Flatten())
  model.add(Dense(64))
  model.add(Activation('tanh'))

  model.add(Dense(5))
  model.add(Activation('sigmoid'))

  model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

  return model

def train_model(train_set):
  filepath = '../models/CNN_model.hd5'
  checkpoint = ModelCheckpoint(filepath, monitor = 'accuracy', verbose = 1, save_best_only = True, save_weights_only = False, mode = 'auto')
  callback_lists = [checkpoint]
  model = build_model()
  model.fit_generator(train_set, steps_per_epoch = 50, epochs = 50, callbacks = callback_lists)
  print(model.summary())

def classify(test_set, y_test):
  pred_model = load_model('../models/CNN_model.hd5')
  preds = pred_model.predict_generator(test_set)
  y_pred =  np.argmax(preds, axis=1)
  print(accuracy_score(y_test, y_pred))


filepath = "../input/neutral/test"

# oPath = save_labels(filepath)
# create_patch(filepath, oPath, 15)

dg = ImageDataGenerator()
train_set = dg.flow_from_directory('../output/patches/train', class_mode = 'categorical', target_size = (30,30))
test_set = dg.flow_from_directory('../output/patches/test', class_mode = 'categorical', target_size = (30,30), batch_size = 92)

y_test_tmp = np.zeros((460,5))
counter = 0
for i in range(len(test_set)):
  y_test_tmp[counter:counter+92,:] = test_set[i][1]
  counter+=92

y_test = np.argwhere(y_test_tmp==1)[:,1]
# train_model(train_set)
classify(test_set, y_test)





