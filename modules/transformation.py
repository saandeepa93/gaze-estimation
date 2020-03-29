import numpy as np
import cv2
import os


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
    P = np.identity(3)

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
    # Xt[[2,3]] = Xt[[3,2]]

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