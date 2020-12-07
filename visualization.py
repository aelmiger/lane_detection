"""

 Title: Visualization Class
 Author: Anton Elmiger
 Created: 2020-05-21

 Information: Class to visualize the opencv images

"""
import numpy as np
import cv2
import constants as const
import math


class Visualization:
        def __init__(self):
            self.array_of_imgs = []
            self.vidArray = []
        
        def clear_imgs(self):
            self.array_of_imgs = []

        def append_img(self,img):
            res_img = cv2.resize(img, (640, 480))
            if len(img.shape) == 2:
                res_img = cv2.cvtColor(res_img, cv2.COLOR_GRAY2RGB)
            self.array_of_imgs.append(res_img)

        def draw_lane_lines(self,img,lane_class):
            lane_c = lane_class.hyperbola_pair(lane_class.bc)
            lane_r = lane_class.hyperbola_pair(lane_class.br)
            lane_l = lane_class.hyperbola_pair(lane_class.bl)

            draw_points_c = (np.asarray([lane_c, lane_class.v]).T).astype(np.int32)
            draw_points_l = (np.asarray([lane_l, lane_class.v]).T).astype(np.int32)
            draw_points_r = (np.asarray([lane_r, lane_class.v]).T).astype(np.int32)

            cv2.polylines(img, [draw_points_c[int(max(const.HORIZON+20,30)):]], False, (0,255,0),10)
            cv2.polylines(img, [draw_points_l[int(max(const.HORIZON+20,30)):]], False, (255,0,0),4)
            cv2.polylines(img, [draw_points_r[int(max(const.HORIZON+20,30)):]], False, (0,0,255),4) 
            return img
        
        def draw_lane_points(self,img,lane_class):
            draw_points_l = (np.asarray([lane_class.left_lane_points.reshape(-1,), lane_class.v]).T).astype(np.int32)
            for point in draw_points_l:
                img = cv2.circle(img,tuple(point),8,(255,0,0),-1)

            draw_points_r = (np.asarray([lane_class.right_lane_points.reshape(-1,), lane_class.v]).T).astype(np.int32)
            for point in draw_points_r:
                img = cv2.circle(img,tuple(point),8,(0,0,255),-1)
                            

        def show_imgs(self):
            n_images = len(self.array_of_imgs)
            rows = math.floor(n_images/3)+1
            row_imgs = self.array_of_imgs[0]
            for i in range(n_images-1):
                row_imgs = np.hstack((row_imgs, self.array_of_imgs[i+1]))
            cv2.imshow("Imgs", row_imgs)
            cv2.waitKey(1)
            if const.WRITE_VIDEO:
                self.vidArray.append(row_imgs)

        def write_video(self):
            height, width, layers = self.vidArray[0].shape
            size = (width,height)

            out = cv2.VideoWriter('vidOut.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
            for i in range(len(self.vidArray)):
                out.write(self.vidArray[i])
            out.release()


