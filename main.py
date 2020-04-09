import numpy as np
import joblib

from modules.onemillisecond import OneMS
from modules.preprocess import preprocessing
from modules.triplet_init import init, load_train


def step_1():
  # pre-processing
  preprocessing( './input/raw/train/', './input/train/')


def step_2():
  # raw to processed data
  filepath = './input/train/'
  triplet = np.array(init(5, 447, filepath))
  print(triplet.shape)
  np.save('./input/train/triplet/triplets.npy', triplet)


def one_millisecond():
  mean_landmarks = np.load('./input/train/frontal/landmark_mean.npy')
  X_train = load_train('./input/train//triplet/triplets.npy', 446)
  # oneMS = OneMS(mean_landmarks, eta = 0.3, T = 10, K = 500, d = 5, t_count = 500)
  oneMS = OneMS(mean_landmarks, eta = 0.3, T = 1, K = 1, d = 5, t_count = 20)
  oneMS.fit(X_train)
  oneMS.print()
  joblib.dump(oneMS, './models/best_model.npy')

def predict(path):
  import cv2
  import dlib
  img = cv2.imread(path)
  print(img.shape)
  detector = dlib.get_frontal_face_detector()
  m = joblib.load('./models/best_model.npy')
  dets = detector(img, 1)
  for d in dets:
    xmin, ymin = d.left(), d.top()
    xmax, ymax = d.right(), d.bottom()

  cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
  lms = m.predict(img).astype(np.int)
  print(lms)
  lms += np.array([xmin, ymin])
  print(lms)
  for i in lms:
    cv2.circle(img, (i[0],i[1]), 3, (0, 0, 255), 2)
  cv2.imshow("image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def main():
  pass
  # step_1()
  # step_2()
  # one_millisecond()
  # m.print()
  # predict('./input/raw/train/Sub31_vid_1_frame26.png')


if __name__ == '__main__':
  main()
