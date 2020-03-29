import cv2
import dlib
import argparse
import time

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image',default = './input/rotated/frame194.png', help='path to image file')
args = ap.parse_args()

image = cv2.imread(args.image)

if image is None:
    print("Could not read input image")
    exit()



def show_cnn_faces(image):

    # initialize hog + svm based face detector
  hog_face_detector = dlib.get_frontal_face_detector()

  # initialize cnn based face detector with the weights
  cnn_face_detector = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')


  start = time.time()

  # apply face detection (hog)--------------------------------------------------
  faces_hog = hog_face_detector(image, 1)

  end = time.time()
  print("Execution Time (in seconds) :")
  print("HOG : ", format(end - start, '.2f'))

  # loop over detected faces
  for face in faces_hog:
      x = face.left()
      y = face.top()
      w = face.right() - x
      h = face.bottom() - y

      # draw box over face
      cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)



  start = time.time()


  # apply face detection (cnn)-----------------------------------
  faces_cnn = cnn_face_detector(image, 1)

  end = time.time()
  print("CNN : ", format(end - start, '.2f'))

  # loop over detected faces
  for face in faces_cnn:
      x = face.rect.left()
      y = face.rect.top()
      w = face.rect.right() - x
      h = face.rect.bottom() - y

      # draw box over face
      cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)


  img_height, img_width = image.shape[:2]

  cv2.putText(image, "HOG", (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                  (0,255,0), 2)

  cv2.putText(image, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                  (0,0,255), 2)

  # display output image
  cv2.imshow("face detection with dlib", image)


def bootstrap(flag):
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

bootstrap(0)
