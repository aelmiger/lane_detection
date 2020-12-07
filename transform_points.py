"""

 Title: Transform Points
 Author: Anton Elmiger
 Created: 2020-05-26

 Information: Class used to transform Points between camera space and world
              space
"""
import numpy as np
import constants as const
from scipy.spatial.transform import Rotation as R


class Transform_Points:
    def __init__(self,_cam_angle=45):
        # cam angle from horizontal down in [degrees]]
        cam_angle = _cam_angle

        self.rot_mat_inv = R.from_euler(
            'xyz', [-90-cam_angle, 0, -90], degrees=True).as_matrix()
        self.rot_mat = np.linalg.inv(self.rot_mat_inv)

        # fov = 160 #fov in degrees/2
        width = const.IMAGE_WIDTH
        height = const.IMAGE_HEIGHT
        aspect_ratio = width / height
        fx = 417/2  # width / (np.tan(np.radians(fov) / 2) * 2)
        fy = fx
        self.cameraMatrix = np.array(
            [[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
        self.cameraMatrixInv = np.linalg.inv(self.cameraMatrix)
        self.tt = -np.array([[0.182], [0.], [0.195]])
        self.rotationMatrix = self.rot_mat  # np.empty([3, 3])
        self.tvec = self.rotationMatrix @ self.tt
        self.rotationMatrixInv = np.linalg.inv(self.rotationMatrix)

        self.poly_koeff = np.array([0.0, 0.0, 0.0])

    def imagePoint_to_worldPoint(self, imgPoints):
        imgPoints = imgPoints.T
        n, m = imgPoints.shape

        imgPoints = np.vstack([imgPoints, np.ones((1, m))])
        leftSideMat = self.rotationMatrixInv.dot(
            self.cameraMatrixInv).dot(imgPoints)
        rightSideMat = self.rotationMatrixInv.dot(self.tvec)
        s = (0 + rightSideMat[2, 0])/leftSideMat[2, :]
        return self.rotationMatrixInv.dot(s*self.cameraMatrixInv.dot(imgPoints)-self.tvec)

    def worldPoint_to_imagePoint(self, worldPoint):
        worldPoint = worldPoint.reshape(-1, 1)
        rightSideMat = self.cameraMatrix.dot(
            self.rotationMatrix.dot(worldPoint)+self.tvec)
        return np.round((rightSideMat/rightSideMat[2, 0])[0:2])

    def transform_lane_to_poly(self, lane_class):
        lane_points = np.hstack((lane_class.lane.reshape(
            (-1, 1)), lane_class.v.reshape((-1, 1))))
        lane_points = lane_points[int(max(const.HORIZON+20, 20)):, :]
        worldCoord = self.imagePoint_to_worldPoint(lane_points).T
        self.poly_koeff = np.polyfit(worldCoord[:, 0], worldCoord[:, 1], 2)

    def send_lane_tcp(self, conn):
        conn.sendall(self.poly_koeff.astype(np.float64).tobytes())
