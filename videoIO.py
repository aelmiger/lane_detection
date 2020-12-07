"""

 Title: Camera IO Class
 Author: Anton Elmiger
 Created: 2020-05-21

 Information: Class to handle the video input in a threaded manner

"""

import cv2
import numpy as np

import os
import time
import threading

import preprocessing as pre
import constants as const

class VideoStream:
    def __init__(self):
        self.frame = None
        self.retrieved = False
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.new_frame = threading.Event()
        self.running = False
        self.ppData, self.maps = pre.loadPPData(const.PATH+'data')


        # Setting up the video stream
        if const.MODE == "camera":
            self.vs = cv2.VideoCapture(0) # create Gstreamer pipeline
            self.retrieved, self.frame = self.vs.read()                         # read first frame to init vid stream
        elif const.MODE == "video":
            self.vs = cv2.VideoCapture(const.VIDEO_FILE)
            self.retrieved, self.frame = self.vs.read()
        elif const.MODE == "simulink":
            pass
        else:
            assert 0, "No video mode defined"


    # Starting the camera thread
    def start(self):
        if const.MODE == "camera":                          # Only start a thread if a camera is used
            if self.vs != None:
                self.running = True
                self.read_thread = threading.Thread(target=self.update)
                self.read_thread.daemon = True
                self.read_thread.start()
        return self

    # Retrieving camera frame
    def update(self):
        while self.running:
            retrieved, frame = self.vs.read()               # Read GStreamer class
            with self.read_lock:                            # Only write to class variable with read lock to prevent corruption
                self.retrieved = retrieved
                self.frame = frame
                self.new_frame.set()                        # Set the new_frame event to prevent read function from reading same frame multiple times

    # Reading camera frame
    def read(self):
        if const.MODE == "camera":
            with self.read_lock:
                frame = self.frame.copy()
                retrieved = self.retrieved
                self.new_frame.clear()
            return frame, retrieved

        elif const.MODE == "video":
            retrieved, frame = self.vs.read()
            return frame, retrieved

        elif const.MODE == "simulink":
            frame = self.read_img_from_sim()
            return frame,True

    # undistort lens distortion from camera image
    def undistort(self,img):
        return cv2.remap(img,self.maps[0],self.maps[1],cv2.INTER_LINEAR)

    # birds eye view transformetion of image
    def bev(self,img):
        bevImg = cv2.resize(img, (1280, 720))
        return cv2.warpPerspective(bevImg,self.ppData[2],(640,480))

    # inverse birds eye view transformation
    def bevInv(self,img):
        img = cv2.resize(img, (640, 480))
        return cv2.warpPerspective(img,np.linalg.inv(self.ppData[2]),(1280,720))


    # Ending video stream
    def release(self):
        if self.vs != None:
            self.vs.release()
            self.running = False
            self.vs = None
        if self.read_thread != None:
            self.read_thread.join()
    
    # function searches for the simulink image, loads it and then deletes it
    def read_img_from_sim(self):
        length = 230400
        data = b''
        while len(data) < length:
            # doing it in batches is generally better than trying
            # to do it all in one go, so I believe.
            to_read = length - len(data)
            data += self.conn.recv(4096 if to_read > 4096 else to_read)

        img = np.array(list(data))
        img = img.astype(np.uint8)
        img = img.reshape((360, 640), order='F')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img



# function to define a gstreamer pipeline (camera connection)
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    # gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
    # Defaults to 1280x720 @ 60fps
    # Flip the image by setting the flip_method (most common values: 0 and 2)
    # display_width and display_height determine the size of the window on the screen
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
