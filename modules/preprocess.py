import numpy as np
import cv2
import os
import sys
import glob
import dlib


detector = dlib.get_frontal_face_detector()
predictor = \
  dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")


def preprocessing(filepath, dest):
  def get_landmarks(frame):
    landmark = np.zeros((68, 2))
    frontal = np.zeros((2,2))
    cnt = 0
    for d,shape in [(d, predictor(frame, d)) for _,d in \
      enumerate(detector(frame, 1))]:
      for i in range(shape.num_parts):
          landmark[i,:] = np.array([shape.part(i).x, shape.part(i).y]) - \
            np.array([d.left(), d.top()])
          landmark[i,0] /= (d.right() - d.left())
          landmark[i,1] /= (d.bottom() - d.top())
      frontal = np.array([[d.left(), d.top()],[d.right(), d.bottom()]])
    return landmark, frontal

  tot = len(glob.glob1(filepath,'*.png'))
  landmarks = np.zeros((tot, 68, 2))
  triplet = np.zeros((tot, ))
  frontals = np.zeros((tot, 2, 2))
  cnt = 0
  for fname in glob.glob(os.path.join(filepath,"*.png")):
    sub = fname.split('/')[-1]
    img = cv2.imread(fname)
    landmarks[cnt,:,:], frontals[cnt,:,:] = get_landmarks(img)
    dob = np.array([(img, landmarks[cnt])])
    np.save(dest+sub, dob)
    cnt+=1
  landmarks_mean = np.average(landmarks, axis = 0)
  frontals_mean = np.average(frontals, axis = 0).astype(np.int)
  np.save('./input/train/frontal/frontal.npy', frontals)
  np.save('./input/train/frontal/frontal_mean.npy', frontals_mean)
  np.save('./input/train/landmark_mean.npy', landmarks_mean)
