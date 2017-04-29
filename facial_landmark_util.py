"""
This file contains the facial landmark detector class and other utility functions regarding facial landmarks.
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

class FacialLandmarkDetector(object):
    def __init__(self, screen_width, screen_height, facial_landmark_predictor_width, path="shape_predictor_68_face_landmarks.dat"):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.facial_landmark_predictor_width = facial_landmark_predictor_width
        self.resize_ratio = (float(self.screen_width) / self.facial_landmark_predictor_width)

    def get_features(self, surface):
        pil_string_image = pygame.image.tostring(surface, "RGBA", False)
        image = np.asarray(Image.frombytes("RGBA", (self.screen_width, self.screen_height), pil_string_image))
        # The image needs to be resized to speed things up.
        if self.screen_width != self.facial_landmark_predictor_width:
            image = imutils.resize(image, width=self.facial_landmark_predictor_width)
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

        return face_coordinates, facial_features


def get_mouth_open_degree(facial_features):
    """

    :param facial_features: format: numpy array of shape (68, 2)
    :return: The degree of mouth opening
    """
    assert isinstance(facial_features,np.ndarray)
    # Get point 62, 64, 66, 68. Don't forget their index is one minus that.
    w = float(abs(facial_features[61,0] - facial_features[63,0]) + abs(facial_features[65,0] - facial_features[67,0])) / 2.0
    h = float(abs(facial_features[61,1] - facial_features[67,1]) + abs(facial_features[63,1] - facial_features[65,1])) / 2.0
    return h / w

def get_mouth_left_corner_ratio(facial_features):
    """

    :param facial_features:
    :return:
    """

    assert isinstance(facial_features,np.ndarray)
    # Get point 62, 64, 66, 68. Don't forget their index is one minus that.
    w = float(abs(facial_features[61,0] - facial_features[63,0]) + abs(facial_features[65,0] - facial_features[67,0])) / 2.0
    h = float(abs(facial_features[61,1] - facial_features[67,1]) + abs(facial_features[63,1] - facial_features[65,1])) / 2.0
    return h / w