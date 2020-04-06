import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import dlib
import skimage.transform as tf

def show_img(img):
  cv2.imshow("image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def denormalize(lms, mean_frontal):
    lms[:,0] *= ((mean_frontal[1][0] - mean_frontal[0][0]))
    lms[:,1] *= ((mean_frontal[1][1] - mean_frontal[0][1]))
    lms += mean_frontal[0]
    return lms.astype(np.int)



detector = dlib.get_frontal_face_detector()
img = cv2.imread('./input/raw/train/frame1.png')
rows,cols,ch = img.shape


dets = detector(img, 1)
for k, d in enumerate(dets):
  frontal = np.array([[d.left(), d.top()], [d.right(), d.bottom()]])

mean_landmarks = np.load('./input/train/landmark_mean.npy')
mean_frontal = np.load('./input/train/frontal/frontal_mean.npy')
frame_1 = np.load('./input/train/frame1.png.npy', allow_pickle = True)
triplet = np.load('./input/train/triplets.npy', allow_pickle = True)

mean_landmarks = denormalize(mean_landmarks, mean_frontal)
img_dlib = denormalize(triplet[0][0][1], mean_frontal)
frame1 = denormalize(frame_1[0][1], frontal)



for i in mean_landmarks:
  cv2.circle(img, (i[0], i[1]), 4, (255, 0, 0), 2)


P = tf.estimate_transform('similarity' ,mean_landmarks, frame1)

img_dlib = np.hstack((img_dlib, np.ones((img_dlib.shape[0], 1))))
print(P.params)

frame_inv = np.matmul(P.params, img_dlib.T).T.astype(np.int)

# for i in frame_inv[:,:-1]:
#   cv2.circle(img, (i[0], i[1]), 4, (0, 0, 255), 2)

print(frame_inv[:,:-1])
print(frame1)

show_img(img)

# dst = cv2.warpAffine(img,M,(cols,rows))

# show_img(dst)


# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()