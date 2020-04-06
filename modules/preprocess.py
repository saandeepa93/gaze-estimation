import numpy as np
import cv2
import os
import glob
import dlib


detector = dlib.get_frontal_face_detector()
predictor = \
  dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")


def denormalize(lms, mean_frontal):
    lms[:,0] *= ((mean_frontal[1][0] - mean_frontal[0][0]))
    lms[:,1] *= ((mean_frontal[1][1] - mean_frontal[0][1]))
    lms += mean_frontal[0]
    return lms.astype(np.int)

def show_img(img, landmarks, mean_frontal):
  landmarks = denormalize(landmarks, mean_frontal)
  for i in landmarks:
    cv2.circle(img, (i[0], i[1]), 4, (255, 0, 0), 2)

  cv2.imshow("image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


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
      frontal = np.array([[d.left(), d.top()],[d.bottom(), d.right()]])
    return landmark, frontal

  tot = len(glob.glob1(filepath,'*.png'))
  landmarks = np.zeros((tot, 68, 2))
  triplet = np.zeros((tot, ))
  frontals = np.zeros((tot, 2, 2))
  cnt = 0
  for fname in glob.glob(os.path.join(filepath,"*.png")):
    print(fname)
    sub = fname.split('/')[-1]
    img = cv2.imread(fname)
    landmarks[cnt,:,:], frontals[cnt,:,:] = get_landmarks(img)
    dob = np.array([(img, landmarks[cnt], frontals[cnt])])
    np.save(dest+sub, dob)
    cnt+=1

  landmarks_mean = np.average(landmarks, axis = 0)
  np.save('./input/train/frontal/frontal.npy', frontals)
  np.save('./input/train//frontal/landmark_mean.npy', landmarks_mean)
