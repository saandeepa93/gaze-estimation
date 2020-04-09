import cv2
import dlib
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt




def show_cnn_faces(image):

  det_flag = 0 #0 is dlib and 1 is CNN
    # initialize hog + svm based face detector
  hog_face_detector = dlib.get_frontal_face_detector()

  dets = hog_face_detector(image, 1)
  if len(dets) == 0:
    print("inside if")
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')
    dets = cnn_face_detector(image, 1)
    if len(dets) == 0:
      print("No face")
      return
    det_flag = 1
    for face in dets:
      x = face.rect.left()
      y = face.rect.top()
      w = face.rect.right() - x
      h = face.rect.bottom() - y

      # draw box over face
      cv2.rectangle(image, (x + 20,y + 20), (x+w+20,y+h+20), (0,0,255), 2)

  else:
    for face in dets:
      x = face.left()
      y = face.top()
      w = face.right() - x
      h = face.bottom() - y

      # draw box over face
      print(f"left: {x}, top:  {y}, right: {face.right()}, bottom:\
        {face.bottom()}")
      cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)


  print(len(dets))


  # initialize cnn based face detector with the weights


  predictor = \
  dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

  if det_flag == 0:
    landmark = np.zeros((68, 2))
    frontal = np.zeros((2,2))
    for d, shape in [(d, predictor(image, d)) for _,d in enumerate(dets)]:
      xmin, xmax, ymin, ymax = d.left(), d.right(), d.top(), d.bottom()
      frontal = np.array([[xmin, ymin], [xmax, ymax]])
      cnt = 0
      for i in range(shape.num_parts):
        lx, ly = shape.part(i).x, shape.part(i).y
        landmark[cnt,:] = np.array([lx, ly])
        # diffx, diffy = lx - xmin, ly - ymin
        # divx, divy = diffx/(xmax-xmin), diffy/(ymax-ymin)
        # cv2.circle(image, (shape.part(i).x, shape.part(i).y), 4, (255, 0, 0), 2)
        cnt+=1

    return landmark, frontal
  else:
    for shape in [predictor(image, d.rect) for _,d in enumerate(dets)]:
        for i in range(shape.num_parts):
          cv2.circle(image, (shape.part(i).x, shape.part(i).y), 4, (0, 0, 255), 2)

  # cv2.imshow("face detection with dlib", image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

def bootstrap(flag):
  if flag == 1:
    img = cv2.imread('./Sub28_vid_1_frame44.png')
    show_cnn_faces(img)
  else:

    video = cv2.VideoCapture(0)
    counter = 0
    while(True):
      counter = (counter + 1) % 100
      if counter % 10 != 0:
        continue

      ret, frame = video.read()
      show_cnn_faces(frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        video.release()
        break

    cv2.destroyAllWindows()

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image',default =\
  './input/raw/train/Sub36_vid_1_frame22.png', help='path to image file')
args = ap.parse_args()

image = cv2.imread(args.image)
print("image size", image.shape)

if image is None:
    print("Could not read input image")
    exit()

landmark, frontal = show_cnn_faces(image)
r, c = 420, 250
image[r-5:r+5, c-5:c+5,:] = (255, 0, 255)
cv2.circle(image, (420, 250), 3, (255, 0, 0), 2)
# cv2.circle(image, (250, 400), 3, (255, 0, 0), 2)
# cv2.circle(image, (250, 400), 3, (255, 0, 0), 2)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()