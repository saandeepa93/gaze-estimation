import numpy as np
import cv2
import os
import sys
from natsort import natsorted

'''
   * Checks if the new points are within the given polygon.
   * Uses the distance calculation from the given lines to determine if it lies above/below is particular line
   * If all the values returned for each line is positive or 0, it is a valid pixel
'''
class Quadrilateral:
    def __init__(self, P):
        self.P = P

        self.F = np.zeros((4,P.shape[0]))
        for i in range(P.shape[0]):
            j = 0 if i==P.shape[0]-1 else i+1
            self.F[0][i] = np.array([[P[j][1]-P[i][1]]])
            self.F[1][i] = np.array([[P[i][0]-P[j][0]]])
            self.F[2][i] = np.array([[-P[i][0] * P[j][1]]])
            self.F[3][i] = np.array([[P[i][1]*P[j][0]]])

    def roi_filter(self, P):
        r, _ = P.shape
        res = np.matmul(np.hstack((P,np.ones((r,2)))), self.F )
        return P[(res >= 0).all(axis = 1)]

    def get_roi(self):
        xmin, ymin = np.amin(self.P[:,0]), np.amin(self.P[:,1])
        xmax, ymax = np.amax(self.P[:,0]), np.amax(self.P[:,1])

        x = np.arange(xmin,xmax)
        y = np.arange(ymin,ymax)

        x = np.repeat(x,ymax-ymin)
        y = np.tile(y,x.shape[0]//y.shape[0])
        w = (np.vstack((x,y)).astype(np.int))

        return self.roi_filter(w.T)

def click_event(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pos_lst = param[0]
        pos_lst.append([x,y])

def get_corner_coordinates(img_original_rgb,params):
    cv2.imshow("image",img_original_rgb)
    cv2.setMouseCallback("image",click_event,params)
    cv2.waitKey()
    cv2.destroyAllWindows

def map_corner_coordinates(img_orig,img_rep,c_pts, mean):
    #Given 4 corner points and their mean, automatically map corner points to corresponding corner location
    bools = np.empty(shape=(4,2))
    bools[:,0] = (c_pts<mean[0])[:,0].astype(float)
    bools[:,1] = (c_pts<mean[1])[:,1].astype(float)
    bools = bools.astype(int).astype(str)
    bools = [''.join(i) for i in bools]
    order = ['11','01','10','00']
    c_pts_target = np.array([c_pts[bools.index(i)] for i in order])
    print(c_pts_target)
    c_pts_source = [[0,0],[img_rep.shape[0],0],[0,img_rep.shape[1]], [img_rep.shape[0],img_rep.shape[1]]]
    return np.array(c_pts_target),np.array(c_pts_source)

'''
    * returns initial P
    * Has been initialized to identity matrix
'''
def get_initial_p(Xs, Xt):
    P1 = np.array([
        [0,0,1,0,0,0],
        [0,0,0,1,0,0],
        [1,0,0,0,0,0]
    ])

    P2 = np.array([
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
        [0,1,0,0,0,0]
    ])

    Xs = np.vstack((Xs.T,np.ones((1,Xs.shape[0]))))
    Xt = np.vstack((Xt.T,np.ones((1,Xt.shape[0]))))

    Z = np.array([[1,0,0],[0,1,0]])
    dX = np.matmul(Z,Xt-Xs).flatten()

    J1 = np.matmul(Xt.T,P1)
    J2 = np.matmul(Xt.T,P2)
    J = np.vstack((J1,J2))

    A = np.matmul(J.T,J)
    b = np.matmul(J.T,dX)

    P = np.pad(np.matmul(np.linalg.inv(A),b),(0,2),'constant')
    P = np.append(P,1).reshape(3,3)
    # P = np.identity(3)

    Xest = np.matmul(Xt.T,P)
    Xest /= Xest[:,-1][:,np.newaxis]
    return P, Xest.T, Xt, Xs

'''
    * Define Initial constant matrices to calculate Hessian(A) and b
    * The optimized matrix operation(Similar to affine) has been derived in the report
'''

def get_J(X,P,Xt):
    P1 = np.array([
        [1,0,0,0,0,0,-1,0],
        [0,1,0,0,0,0,0,-1],
        [0,0,1,0,0,0,0,0]
    ])

    P2 = np.array([
        [0,0,0,1,0,0,-1,0],
        [0,0,0,0,1,0,0,-1],
        [0,0,0,0,0,1,0,0]
    ])

    P11 = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0 ]
    ])

    P22 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0]
    ])

    P[2][2] = 1
    X_est = np.matmul(P,Xt)
    D = X_est[-1, :][:, np.newaxis]
    X_est /= D.T
    J1 = (np.matmul(Xt.T,P1) * np.matmul(X_est.T, P11))/D
    J2 = (np.matmul(Xt.T,P2) * np.matmul(X_est.T, P22))/D
    return np.asarray(np.vstack((J1,J2))) , X_est, P

'''
    * Calculates new homography matrix iteratively
'''
def get_transformation(Xs,X,iterations):
    '''
    Xt is expected to be in clockwise direction
    '''
    Xt = np.copy(X)
    Xt[[2,3]] = Xt[[3,2]]

    P,_, Xt, Xs = get_initial_p(Xs, Xt)

    residual = 10e8
    residual_new = residual-1
    count = iterations
    while (residual_new != 0 and residual_new < residual) or count > 0:
        residual = residual_new
        J,X_est, P = get_J(Xs, P, Xt)

        Z = np.array([[1,0,0],[0,1,0]])
        dR = np.matmul(Z,Xs-X_est).flatten()
        residual_new = int(np.matmul(dR.T, dR))

        A = np.matmul(J.T,J)
        b = np.matmul(J.T,dR)

        dp = np.matmul(np.linalg.inv(A + 0.1 * np.diag(np.diag(A))), b)
        P += np.append(dp,0).reshape(3,3)
        count -= 1

    return P

'''
    * Applies the transorfamtion to all valid pixels
    * Saves the result in output folder
'''

def apply_transformation(P,c_pts_source,c_pts_target, img_target, img_source,output_name):
    r, c, _ = img_target.shape
    r2, c2, _ = img_source.shape

    img_test = np.ones((r,c,3))
    t = np.array([
        [c_pts_target[2][0], c_pts_target[2][1]],
        [c_pts_target[3][0], c_pts_target[3][1]],
        [c_pts_target[1][0], c_pts_target[1][1]],
        [c_pts_target[0][0], c_pts_target[0][1]]
    ])

    q2 = Quadrilateral(c_pts_target[::-1])
    w2 = q2.get_roi()
    print(w2.shape)
    today = np.matmul(P , np.hstack((w2, np.ones((w2.shape[0],1)))).T)
    today = (today[:-1, :] / today[-1, :]).astype(np.int)
    today[today < 0] = 0
    today[0, :][today[0, :] > r2 - 1] = r2 - 1
    today[1, :][today[1, :] > c2 - 1] = c2 - 1

    img_target[w2[:,1],w2[:,0]] =img_source[today[1,:],today[0,:]]
    cv2.imwrite(f".\\Output\{output_name}",img_target)
    cv2.waitKey()
    cv2.destroyAllWindows

def create_video(image_lst,video_name,filepath):
    print(os.path.join(filepath, image_lst[0]))
    frame = cv2.imread(os.path.join(filepath, image_lst[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in image_lst:
        video.write(cv2.imread(os.path.join(filepath, image)))

    cv2.destroyAllWindows()
    video.release()

def sequential_transformation(filepath,c_pts_target,\
                img_original_rgb,output_path,iterations,c_pts_transformed):
    #Read image
    img_rep = cv2.resize(cv2.imread(filepath),(600,600))

    c_pts_source = np.array([[0,0],[img_rep.shape[0],0],[0,img_rep.shape[1]], [img_rep.shape[0],img_rep.shape[1]]])
    #map_corner_coordinates(img_original,img_rep,np.array(pos_lst), list(np.mean(pos_lst,0)))

    '''get final homography matrix'''
    P = get_transformation(c_pts_source,c_pts_transformed,iterations)

    '''Apply transformation using final homography matrix'''
    apply_transformation(P,c_pts_source,c_pts_target,img_original_rgb,img_rep,output_path)


def main():
    params = list()
    pos_lst = []
    params.append(pos_lst)

    '''Get command line arguments'''
    filename_orig = 'stopsign3.jpg' #str(sys.argv[1])
    filename_rep =  'cokeBig.png' #str(sys.argv[2])
    iterations = 100 #int(sys.argv[3])
    output_name = 'test' #str(sys.argv[4])
    video_flag = 0 #int(sys.argv[5])
    sequence_folder = 'moon_frames'#str(sys.argv[6])
    n_pts = 4 #float(sys.argv[5])

    '''Load the target images'''
    filepath = os.getcwd() + "\Input\\"+filename_orig
    img_original_rgb = cv2.resize(cv2.imread(filepath),(600,600))
    img_original = cv2.cvtColor(img_original_rgb,cv2.COLOR_BGR2GRAY)

    '''Select corner points of the image'''
    get_corner_coordinates(img_original_rgb,params)

    '''Map selected points to corner'''
    c_pts_target = np.array(pos_lst)
    c_pts_transformed = enclosing.get_4_from_n_point(c_pts_target) if len(pos_lst)>4 else c_pts_target

    '''Video Mode transformation'''
    if video_flag==1:
        count = 0
        image_lst = []
        video_name = 'video.avi'

        filepath = os.getcwd()+"\Input\\" + sequence_folder

        #Create directory if folder does not exist
        if not os.path.isdir(os.getcwd()+"\Output\\" + sequence_folder):
            os.mkdir(os.getcwd()+"\Output\\" + sequence_folder)
            print("created ",sequence_folder)

        #loop through each frame
        for fname in natsorted(os.listdir(filepath)):
            image_lst.append(fname)
            sequential_transformation(filepath+"\\"+fname,\
                                     c_pts_target,img_original_rgb,sequence_folder+"\\"+str(count)+".png",iterations)
            count+=1

        #Create a video of the frames
        create_video(image_lst,video_name,os.getcwd()+"\Output\\" +\
             sequence_folder)

    else:
        filepath = os.getcwd()+"\Input\\" + filename_rep
        sequential_transformation(filepath,\
                                     c_pts_target,img_original_rgb,output_name+".png",iterations,c_pts_transformed)


if __name__ == '__main__':
    main()