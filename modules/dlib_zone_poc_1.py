import numpy as np
from numpy import sin as sin, cos as cos
import cv2
import dlib
import glob
import os
import sys
import face_alignment
from transformation import get_transformation

# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1("./models/mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

def get_features(img):
  dets = detector(img, 1)
  cnn_dets = cnn_face_detector(img,1)
  return [(d, predictor(img, d)) for _,d in enumerate(dets)], [(d, predictor(img, d)) for _,d in enumerate(cnn_dets)]

def get_landmarks(frame):
  dd, dd2 = get_features(frame)
  a = 0 # roll 0 - 180
  b = 0 # yaw 0 - 180
  c = 0 # pitch 0 - 180
  for d,shape in dd:
    temp = []
    temp_2 = []
    temp_3 = []

    indices = [28, 8]
    indices_2 = [3, 34, 13]
    indices_3 = [28, 34, 8]
    for i in range(68):
      if i in indices:
        temp.append([shape.part(i).x,shape.part(i).y])
        cv2.circle(frame, (shape.part(i).x,shape.part(i).y), 2, (255,0,0), 1)
      if i in indices_2:
        temp_2.append([shape.part(i).x,shape.part(i).y])
        cv2.circle(frame, (shape.part(i).x,shape.part(i).y), 2, (255,0,0), 1)
      if i in indices_3:
        temp_3.append([shape.part(i).x,shape.part(i).y])
        cv2.circle(frame, (shape.part(i).x,shape.part(i).y), 2, (255,0,0), 1)

    temp = np.array(temp)
    temp[:,0] -= temp[0][0]
    temp[:,1] -= temp[0][1]

    temp_2 = np.array(temp_2)[:,0]
    temp_2[:] -= temp_2[-1]

    temp_3 = np.array(temp_3)[:,1]
    temp_3[:] -= temp_3[-1]

    m, n = temp_2[-1] - temp_2[1], temp_2[-1] - temp_2[0]
    b = np.abs((np.arctan(m/n) * 180 / np.pi) * 2).astype(np.int)

    o, p = temp_3[-1] - temp_3[1], temp_3[-1] - temp_3[0]
    c = np.abs((np.arctan(o/p) * 180 / np.pi) * 4).astype(np.int)

    a = (np.arctan((temp[0][1] - temp[1][1]) / (temp[0][0] - temp[1][0])) * 180 / np.pi)
    for i in range(shape.num_parts):
      cv2.circle(frame, (shape.part(i).x,shape.part(i).y), 5, (0,0,255), 2)

  cv2.putText(frame, f'Roll: {a}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
  cv2.putText(frame, f'Yaw {b}', (50, 75), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
  cv2.putText(frame, f'Pitch {c}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

  col = 0
  if b <= 60:
    col = 'left'
  elif b > 60 and b < 120:
    col = 'square'
  else:
    col = 'right'

  row = 0
  if c <= 60:
    row = 'up'
  elif c > 60 and c < 120:
    row = 'square'
  else:
    row = 'down'

  cv2.line(frame, (0, frame.shape[0]//3), (frame.shape[1], frame.shape[0]//3), (255, 0, 0), 2)
  cv2.line(frame, (0, 2 * frame.shape[0]//3), (frame.shape[1], 2 * frame.shape[0]//3), (255, 0, 0), 2)

  cv2.line(frame, (frame.shape[1]//3, 0), (frame.shape[1]//3, frame.shape[1]), (255, 0, 0), 2)
  cv2.line(frame, (2 * frame.shape[1]//3, 0), (2 * frame.shape[1]//3, frame.shape[1]), (255, 0, 0), 2)

  cv2.putText(frame, f'Column {col}', (50, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
  cv2.putText(frame, f'Row {row}', (50, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
  cv2.imshow('frame', frame)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  return dd

def get_rotated_dd(filepath_rot):
  lm_rot = np.array((68,2))
  img_rot = cv2.imread(filepath_rot)
  dd_rot = get_landmarks(img_rot)
  lm_rot = np.array([[np.array([shape.part(i).x,shape.part(i).y]) for i in range(shape.num_parts)] for _,shape in dd_rot]).reshape((68,2))
  return lm_rot


def bootstrap(flag=True):
  landmarks = np.zeros((38,68,2))
  if flag:
    cnt = 0
    filepath = "./input/neutral"
    filepath_rot = "./input/rotated/frame228.png"
    rotated_lm = get_rotated_dd(filepath_rot)
    for fname in glob.glob(os.path.join(filepath,"*.png")):
      frame = cv2.imread(fname)
      dd = get_landmarks(frame)
      landmarks[cnt,:,:] = np.array([[np.array([shape.part(i).x,shape.part(i).y]) for i in range(shape.num_parts)] for _,shape in dd]).reshape((68,2))
      landmarks[cnt,:,:] -= np.average(landmarks[cnt,:,:],axis=0)
      cnt+=1


    neutral_lm = np.average(landmarks, axis = 0)
    xmin, xmax, ymin, ymax = np.amin(neutral_lm[:,0]), np.amax(neutral_lm[:,0]), np.amin(neutral_lm[:,1]), np.amax(neutral_lm[:,1])

    neutral_lm += np.array([xmax + 50, ymax + 50])
    neutral_lm  = neutral_lm.astype(np.int)
    centroid = np.average(neutral_lm, axis = 0).astype(np.int)
    temp = np.ones(((2*xmax+500).astype(np.int), (2*ymax+500).astype(np.int), 3))


    φ = (0 * np.pi/180)  #x, pitch, phi
    θ = (0 * np.pi/180)  #y, yawwwwwww, theta
    ψ = (40 * np.pi/180)   #z, roll, psi

    rotation = np.array([
      [cos(ψ) * cos(θ), - sin(ψ) * cos(φ)],
      [sin(ψ) * cos(θ), cos(ψ) * cos(φ)]
    ])

    neutral_lm_1 = np.matmul(neutral_lm, rotation).astype(np.int)

    rotation = np.array([
      [cos(ψ) * cos(θ), cos(ψ) * sin(θ) * sin(φ) - sin(ψ) * cos(φ), cos(ψ) * sin(θ) * cos(φ) + sin(ψ) * sin(φ)],
      [sin(ψ) * cos(θ), sin(ψ) * sin(θ) * sin(φ) + cos(ψ) * cos(φ), sin(ψ) * sin(θ) *  cos(φ) - cos(ψ) * sin(φ)],
      [-sin(θ), cos(θ) * sin(φ), cos(θ) * cos(φ)]
    ])


    neutral_lm_2 = np.matmul(np.hstack((neutral_lm, np.ones((neutral_lm.shape[0], 1)))), rotation)[:,:2]
    neutral_lm_2 = neutral_lm_2.astype(np.int)
    rot_centroid = np.average(neutral_lm_2, axis = 0).astype(np.int)
    neutral_lm_2 = neutral_lm_2 + (centroid - rot_centroid)
    rot_centroid_1 = np.average(neutral_lm_1, axis = 0).astype(np.int)
    neutral_lm_1 = neutral_lm_1 + (centroid - rot_centroid_1)
    P = get_transformation(neutral_lm, rotated_lm, 50)
    temp_neutral = np.matmul(np.hstack((neutral_lm, np.ones((neutral_lm.shape[0], 1)))), P)
    w, v = np.linalg.eig(P)
    print(w.shape, v.shape)
    # sys.exit(0)
    # print((temp_neutral.shape)
    # temp_neutral = np.divide(temp_neutral,(temp_neutral[:,2][:,np.newaxis]))
    temp_neutral[:,0] = temp_neutral[:,0]  / np.amax(temp_neutral[:,0]) * 2 * xmax
    temp_neutral[:,1] = temp_neutral[:,1]  / np.amax(temp_neutral[:,1]) * ymax

    ss = np.matmul(np.hstack((neutral_lm_2, np.ones((neutral_lm_2.shape[0], 1)))), np.linalg.inv(P))
    ss[:,0] = ss[:,0] / np.amax(ss[:,0]) * xmax
    ss[:,1] = ss[:,1] / np.amax(ss[:,1]) * ymax
    ss = ss[:,:2].astype(np.int)
    neutral_lm = temp_neutral[:,:2].astype(np.int)
    indices = [28, 29, 30, 31, 58, 9, 63, 52]
    for i in range(68):
      if i in indices:
        cv2.circle(temp, (neutral_lm_2[i][0], neutral_lm_2[i][1]), 2, (255,0,0), 1)

    # for pts in (neutral_lm_2 + np.array([200,0])):
      # cv2.circle(temp, (pts[0], pts[1]), 2, (255,0,0), 1)
    # for pts in (neutral_lm_1 + np.array([400,0])):
    #   cv2.circle(temp, (pts[0], pts[1]), 2, (0,0,255), 1)
    cv2.imshow("img", temp)
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
bootstrap(0)