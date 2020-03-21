import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

def get_features(img):
  dets = detector(img, 1)
  return [(d, predictor(img, d)) for _,d in enumerate(dets)]

def get_landmarks(frame):
  dd = get_features(frame)
  for d,shape in dd:
    for i in range(shape.num_parts):
      cv2.circle(frame, (shape.part(i).x,shape.part(i).y), 5, (0,0,255), 2)
  cv2.imshow('frame', frame)
  return dd

def bootstrap(flag=True):
  if flag:
    filepath = "./input/Sub185_2/1/frame50.png"
    frame = cv2.imread(filepath)
    get_landmarks(frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  else:
    video = cv2.VideoCapture(0)
    counter = 0
    while(True):
      counter = (counter + 1) % 100
      if counter % 10 != 0:
        continue

      ret, frame = video.read()
      get_landmarks(frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        video.release()
        break

    cv2.destroyAllWindows()

bootstrap(1)