"""

 Title: Kalman Filter
 Author: Anton Elmiger
 Created: 2020-05-27

 Information: Kalman filtering for the lane model

"""

import numpy as np


class KalmanFilter:
    def __init__(self, kalmanMeasurementNoiseCov=0.1):
        self.R = 1.0 * kalmanMeasurementNoiseCov * np.eye(72*2+1, dtype=float)
        self.x = np.array(
            [[0.], [-0.7], [0.7], [0.], [-80.]], dtype=float)
        self.P = np.array([[100,0,0,0,0],[0,0.05**2,0.9*0.05**2,0,0],[0,0.9*0.05**2,0.05**2,0,0],[0,0,0,1,0],[0,0,0,0,0.01]])*0.1
        self.Q = self.P
    def hyperbola_pair(self, b,v):
        return self.x[0]/(v-self.x[4])+b*(v-self.x[4])+self.x[3]

    # prediction calculates the next expected states
    def predict(self, v, h, k, bl, br, c):
        v = v[0::10]
        v = v.reshape((len(v), 1))
        topMat = np.hstack((1/(v-h), v-h, np.zeros(v.shape),
                            np.ones(v.shape), k/(v-h)**2-bl))
        botMat = np.hstack((1/(v-h), np.zeros(v.shape), v-h,
                            np.ones(v.shape), k/(v-h)**2-br))

        self.H = np.vstack((topMat, botMat, np.array([0, -1, 1, 0, 0])))
        # self.x = np.array([k, bl, br, c, h])

    # update combines measurement with the prediction
    def update(self,ul,ur,l,v):
        ul = ul[0::10]
        ur = ur[0::10]
        v = v[0::10]
        self.P = self.Q + self.P
        # self.s = np.dot(self.hMatrix,
        #                 np.dot(self.predictedCovMatrix,
        #                        np.transpose(self.hMatrix))) + self.measurementNoiseCov

        self.K = np.linalg.inv(self.H.T.dot(np.linalg.inv(self.R)).dot(
            self.H) + np.linalg.inv(self.P)).dot(self.H.T).dot(np.linalg.inv(self.R))

        # self.kalmanGain = np.dot(np.dot(self.predictedCovMatrix, np.transpose(self.hMatrix)),
        #                          np.linalg.pinv(self.s))
        z = np.hstack((ul,ur,l)).reshape((2*len(ul)+1,1))
        h = np.hstack((self.hyperbola_pair(self.x[1],v),self.hyperbola_pair(self.x[2],v),l)).reshape((2*len(ul)+1,1))
        self.x = self.x + np.dot(self.K, (z-h))

        self.P = np.dot((np.eye(5) - np.dot(self.K, self.H)),
                                self.P)
