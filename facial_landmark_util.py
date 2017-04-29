"""
This file contains the facial landmark detector class and other utility functions regarding facial landmarks.
In order to use the facial landmark detector please download and extract
http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
at the root folder.
"""

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
from PIL import Image

import pygame
import pygame.camera
from pygame.locals import *

def tuple_to_rectangle(tup):
    if len(tup) != 4:
        raise AttributeError("Incorrect input for tuple_to_rectangle. Input should be (x,y,w,h)")

    left = int(tup[0])
    top = int(tup[1])
    right = int(tup[0] + tup[2])
    bottom = int(tup[1] + tup[3])
    return dlib.rectangle(left=left, top=top, right=right, bottom=bottom)

class FacialLandmarkDetector(object):
    def __init__(self, screen_width, screen_height, facial_landmark_predictor_width, path="shape_predictor_68_face_landmarks.dat"):
        # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
        if not os.path.isfile(path):
            raise IOError("Cannot find the facial landmark data file. Please download it from "
                          "http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks."
                          "dat.bz2 and extract that directly to the root folder of this project.")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.facial_landmark_predictor_width = facial_landmark_predictor_width
        self.resize_ratio = (float(self.screen_width) / self.facial_landmark_predictor_width)

    def get_image_from_surface(self, surface):
        pil_string_image = pygame.image.tostring(surface, "RGBA", False)
        image = np.asarray(Image.frombytes("RGBA", (self.screen_width, self.screen_height), pil_string_image))
        # The image needs to be resized to speed things up.
        if self.screen_width != self.facial_landmark_predictor_width:
            image = imutils.resize(image, width=self.facial_landmark_predictor_width)
        return image

    def get_features(self, surface):
        image = self.get_image_from_surface(surface)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = self.detector(gray, 1)

        face_coordinates = []  # Item format (x, y, w, h)
        facial_features = []  # Item format: numpy array of shape (68, 2)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            shape = shape * self.resize_ratio
            facial_features.append(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            (x, y, w, h) = (x * self.resize_ratio, y * self.resize_ratio, w * self.resize_ratio, h * self.resize_ratio)
            face_coordinates.append((x, y, w, h))

        # TODO: getLargestFaceBoundingBox
        return face_coordinates, facial_features


def get_mouth_open_score(facial_features):
    """

    :param facial_features: format: numpy array of shape (68, 2)
    :return: The degree of mouth opening
    """
    assert isinstance(facial_features,np.ndarray)
    # Get point 62, 64, 66, 68. Don't forget their index is one minus that.
    w = float(abs(facial_features[61,0] - facial_features[63,0]) + abs(facial_features[65,0] - facial_features[67,0])) / 2.0
    h = float(abs(facial_features[61,1] - facial_features[67,1]) + abs(facial_features[63,1] - facial_features[65,1])) / 2.0
    return h / w

def get_mouth_left_corner_score(facial_features):
    """
    TODO: this current implementation is not invariant to face orientation.
    Resources that might be useful: http://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    https://github.com/dougsouza/face-frontalization
    :param facial_features: format: numpy array of shape (68, 2)
    :return: A ratio between (distance from left mouth corner to right face edge) and the face width.
    Larger score means closer to boarder of the face - thus more distorted facial expression.
    """
    raise NotImplementedError("TODO: this current implementation is not invariant to face orientation. ")
    assert isinstance(facial_features,np.ndarray)
    # First get an average width of the face. Average over three pairs of coordinates
    w = float(abs(facial_features[3,0] - facial_features[15,0]) +
              abs(facial_features[4,0] - facial_features[14,0]) +
              abs(facial_features[5,0] - facial_features[13,0])) / 3.0
    # Now get the distance from 4 to 49.
    d = float(abs(facial_features[4,0] - facial_features[49,0]))
    return 1.0 - d / w