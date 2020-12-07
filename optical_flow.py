"""

 Title: Optical Flow
 Author: Anton Elmiger
 Created: 2020-05-30

 Information: Optical flow with the goal to extract vehicle pose from live video

"""

import numpy as np
import cv2


class OpticalFlow():
    def __init__(self):
        self.track_points = None

        self.feature_params = dict(maxCorners=50,
                                   qualityLevel=0.01,
                                   minDistance=7,
                                   blockSize=5)

        self.lk_params = dict(winSize=(5, 5),
                              maxLevel=4,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

        self.outlier_thresh = 0.3

    def pseudoangle(self, dx, dy):
        p = dx/(abs(dx)+abs(dy))  # -1 .. 1 increasing with x
        if dy < 0:
            return 3 + p  # 2 .. 4 increasing with x
        else:
            return 1 - p  # 0 .. 2 decreasing with x

    def rigid_transform_3D(self, A, B):
        assert len(A) == len(B)

        num_rows, num_cols = A.shape

        if num_rows != 2:
            raise Exception(
                "matrix A is not 2xN, it is {}x{}".format(num_rows, num_cols))

        [num_rows, num_cols] = B.shape
        if num_rows != 2:
            raise Exception(
                "matrix B is not 2xN, it is {}x{}".format(num_rows, num_cols))

        # find mean column wise
        centroid_A = np.mean(A, axis=1).reshape((-1, 1))
        centroid_B = np.mean(B, axis=1).reshape((-1, 1))

        # subtract mean
        Am = A - np.tile(centroid_A, (1, num_cols))
        Bm = B - np.tile(centroid_B, (1, num_cols))

        H = Am.dot(np.transpose(Bm))

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T.dot(U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            # print("det(R) < R, reflection detected!, correcting for it ...\n")
            Vt[1, :] *= -1
            R = Vt.T * U.T

        t = -R.dot(centroid_A) + centroid_B

        return R, t

    def init_tracking(self, img):
        self.track_points = cv2.goodFeaturesToTrack(
            img, mask=None, **self.feature_params)
        self.old_img = img.copy()

        ret, mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        self.mask = cv2.erode(mask, kernel, iterations=8)

    def calc_opt_flow(self, img):

        new_track_points, found, err = cv2.calcOpticalFlowPyrLK(
            self.old_img, img, self.track_points, None, **self.lk_params)
        self.track_points_curr = new_track_points[found == 1]
        self.track_points_prev = self.track_points[found == 1]
        self.remove_outliers()

        R, t = self.rigid_transform_3D(
            self.track_points_prev.T, self.track_points_curr.T)
        if t[1] < 0 or np.linalg.norm(t,2) > 70:
            R[:]=np.NaN
            t[:]=np.NaN

        self.track_points = cv2.goodFeaturesToTrack(
            img, mask=self.mask, **self.feature_params)

        self.track_points = np.vstack(
            (self.track_points, self.track_points_curr.reshape(-1, 1, 2)))
        self.old_img = img.copy()

        return R, t

    def remove_outliers(self):
        correct_dir_ind = self.track_points_curr[:,
                                                 1] > self.track_points_prev[:, 1]
        self.track_points_curr = self.track_points_curr[correct_dir_ind]
        self.track_points_prev = self.track_points_prev[correct_dir_ind]
        norm = np.linalg.norm(self.track_points_curr -
                              self.track_points_prev, axis=1)
        median = np.median(norm)
        self.track_points_curr = self.track_points_curr[abs(
            norm-median) < (median * self.outlier_thresh)]
        self.track_points_prev = self.track_points_prev[abs(
            norm-median) < (median * self.outlier_thresh)]
        psAng = np.zeros(len(self.track_points_curr[:, 0]))
        for i in range(len(self.track_points_curr[:, 0])):
            psAng[i] = self.pseudoangle(self.track_points_curr[i, 0]-self.track_points_prev[i, 0],
                                        self.track_points_curr[i, 1]-self.track_points_prev[i, 1])
        medianAng = np.median(psAng)

        self.track_points_curr = self.track_points_curr[abs(
            psAng-medianAng) < (medianAng * self.outlier_thresh)]
        self.track_points_prev = self.track_points_prev[abs(
            psAng-medianAng) < (medianAng * self.outlier_thresh)]

    def draw_trackings(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        color = np.random.randint(0, 255, (500, 3))
        for i, (new, old) in enumerate(zip(self.track_points_curr, self.track_points_prev)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(img, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(img, (a, b), 3, color[i].tolist(), -1)
        return img
