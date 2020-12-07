"""

 Title: Lane Detection Algorithm
 Author: Anton Elmiger
 Created: 2020-05-26

 Information: Class that extracts a lane from an edge image
              and calculates the corresponding hyperbola-pair parameters

              Lane Model is described in this paper https://ieeexplore.ieee.org/abstract/document/1689679
              and in the wiki of github

"""


import numpy as np
import cv2
from scipy.optimize import lsq_linear
import constants as const

class Lane_Detection:
    def __init__(self):
        self.v = np.arange(0, const.IMAGE_HEIGHT, 1)  # vertical points
        self.u = np.arange(0, const.IMAGE_WIDTH, 1)  # horizontal points

        self.threshold = const.BOUNDARY_THRESH
        self.lane_width = const.LANE_WIDTH

        self.h = const.HORIZON # h-horizon height
        self.k = 0  # k-curvature of lane
        self.bl = 0 # b-skew of left lane
        self.br = 0 # b-skew of right lane
        self.bc = 0 # b-skew of lane center
        self.c = 0 # c-horizontal offset of lane

        self.left_lane_points = np.array([])
        self.right_lane_points = np.array([])
        
        self.lane = np.array([])

        # Bounds for the solving of hyperbola-pair parameters
        # [k,bl,br,c]
        # The constraint on c dramatically increases robustness
        self.low_b = np.array([-500000, -8, -8, const.IMAGE_WIDTH/2 -20])
        self.up_b = np.array([500000, 8, 8, const.IMAGE_WIDTH/2 +20])

    def testImg(self,img):
        print("Test")
        print(img.shape)

    # Calculate lane hyperbola for given parameters
    def hyperbola_pair(self, b):
        return self.k/(self.v-self.h)+b*(self.v-self.h)+self.c

    # Function finds lane points in an edge image and classifies them to left and right lane
    # This function is used if no lane estimate exists, or the estimation is odd
    def get_initial_lane_points(self, edge_image):
        image_height = edge_image.shape[0]
        image_width = edge_image.shape[1]

        # initialize lane arrays
        left_lane_points = np.empty((image_height, 1))
        left_lane_points[:] = np.NAN
        right_lane_points = np.empty((image_height, 1))
        right_lane_points[:] = np.NAN

        lane_numbers = np.arange(image_width)
        edge_image = edge_image / 255

        for row in range(image_height-1, -1, -1):
            curr_row = np.multiply(
                (lane_numbers - image_height), edge_image[row, :])
            points_to_the_right = np.where(curr_row > 0)[0]
            points_to_the_left = np.where(curr_row < 0)[0]
            if points_to_the_right.size > 0:
                right_lane_points[row] = np.amin(points_to_the_right)
            if points_to_the_left.size > 0:
                left_lane_points[row] = np.amax(points_to_the_left)
            if row == 300:
                break
        self.left_lane_points = left_lane_points
        self.right_lane_points = right_lane_points

    # Function finds lane points in an edge image and classifies them to left and right lane
    def lane_points(self, edge_image):
        image_height = edge_image.shape[0]

        # initialize lane arrays
        left_lane_points = np.empty((image_height, 1))
        left_lane_points[:] = np.NAN
        right_lane_points = np.empty((image_height, 1))
        right_lane_points[:] = np.NAN

        # get the "bounding" lanes to filter outliers
        # only points between the bounds are considered inliers
        left_max_bound, left_min_bound, right_max_bound, right_min_bound = self.generate_bounding_lanes()

        # only considere points that are below the horizon (plus some extra space for robustness) if the horizon is in the image
        horizon_index = int(max(self.h+20,0))

        # get the 2D image position of edge pixels that are below the horizon index
        nonzero = cv2.findNonZero(edge_image[horizon_index:]).reshape((-1,2)).T
        # offset the Y-Coordinate by the horizon index
        nonzero[1] += horizon_index

        # classify all points in left bounding area as left lane points
        left_p = nonzero.T[(nonzero[0] < left_max_bound[nonzero[1]]) & (nonzero[0] > left_min_bound[nonzero[1]])]

        # classify all points in right bounding area as left right points
        # the flipping of the array is imortant for the next step
        right_p = np.flipud(nonzero.T[(nonzero[0] < right_max_bound[nonzero[1]]) & (nonzero[0] > right_min_bound[nonzero[1]])])

        # for each vertical row in the image that contains a left lane point ->
        # place the point that is closest the the centerline into the left lane points array
        np.put(left_lane_points,left_p[:,1],left_p[:,0])
        
        # for each vertical row in the image that contains a right lane point ->
        # place the point that is closest the the centerline into the right lane points array
        np.put(right_lane_points,right_p[:,1],right_p[:,0])

        self.left_lane_points = left_lane_points
        self.right_lane_points = right_lane_points


    # Function returns lane lines, that are left and right of the estimated lane lines
    # These bounding lines are then used to define an inlier area
    def generate_bounding_lanes(self):
        # horizontal points left lane
        left_max = self.hyperbola_pair(self.bl+(self.bc-self.bl)/self.threshold)
        # horizontal points left lane
        left_min = self.hyperbola_pair(self.bl-(self.bc-self.bl)/self.threshold)
        # horizontal points left lane
        right_max = self.hyperbola_pair(self.br+(self.bc-self.bl)/self.threshold)
        # horizontal points left lane
        right_min = self.hyperbola_pair(self.br-(self.bc-self.bl)/self.threshold)
        return left_max, left_min, right_max, right_min


    # Function solves for hyperbola-pair lane parameters
    # More info is in the paper listed at the top of this file
    def solve_lane(self):
        l = self.left_lane_points
        r = self.right_lane_points

        # following lines create A matrix  and b vector for least square porblem
        l_ind = ~np.isnan(l)
        r_ind = ~np.isnan(r)
        l_num = l[l_ind]
        r_num = r[r_ind]
        vl = self.v[l_ind.flatten()]
        vr = self.v[r_ind.flatten()]
        l_num = l_num.reshape((len(l_num), 1))
        r_num = r_num.reshape((len(r_num), 1))
        vl = vl.reshape(l_num.shape)
        vr = vr.reshape(r_num.shape)

        lh = (vl-self.h)
        lA = 1/lh
        rh = (vr-self.h)
        rA = 1/rh
        ones = np.ones(l_num.shape)
        zeros = np.zeros(l_num.shape)
        LA = np.hstack((np.hstack((lA, lh)), np.hstack((zeros, ones))))
        ones = np.ones(r_num.shape)
        zeros = np.zeros(r_num.shape)
        RA = np.hstack((np.hstack((rA, zeros)), np.hstack((rh, ones))))
        A = np.vstack((LA, RA))
        b = (np.concatenate((l_num, r_num))).flatten()

        # returning the solved parameters (k,bl,br,c)
        x = lsq_linear(A, b, bounds=(self.low_b, self.up_b), method='bvls', max_iter = 3).x
        # set new lane model param from least square solution
        self.k = x[0]
        self.bl=x[1]
        self.br=x[2]
        self.c = x[3]
        self.bc = (x[1]+x[2])/2
        
        # calc lane points
        self.lane = self.hyperbola_pair(self.bc)

    # function corrects false lane lines or missing lane lines
    def lane_sanity_checks(self,edge_image):
        #lane not found
        if self.k == 0:
            self.get_initial_lane_points(edge_image)
            self.solve_lane()


        #Only one lane found
        if ~np.isfinite(self.left_lane_points).any():
            self.bl = self.br-self.lane_width-0.3
            self.bc = (self.bl+self.br)/2            
        if ~np.isfinite(self.right_lane_points).any():
            self.br = self.bl+self.lane_width+0.3
            self.bc = (self.bl+self.br)/2            

        #Lane width not correct size
        if abs(self.bl-self.br)<(self.lane_width*0.8) or abs(self.bl-self.br)>(self.lane_width)*1.2:
            length_l = np.count_nonzero(~np.isnan(self.left_lane_points))
            length_r = np.count_nonzero(~np.isnan(self.right_lane_points))
            if length_l > length_r:
                self.br = self.bl+self.lane_width
            else:
                self.bl = self.br-self.lane_width
            self.bc = (self.bl+self.br)/2            

        
        #Vehicle not on lane -> recenter lane line
        if self.bc > (self.lane_width/1.1):
            self.bl=self.bl-self.lane_width
            self.br=self.br-self.lane_width
            
        if self.bc < (-self.lane_width/1.1):
            self.bl=self.bl+self.lane_width
            self.br=self.br+self.lane_width

        self.bc = (self.bl+self.br)/2            
        self.lane = self.hyperbola_pair(self.bc)




