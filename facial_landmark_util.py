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
import math
from operator import mul
from PIL import Image
from scipy.spatial import distance as dist


import pygame
import pygame.camera
from pygame.locals import *
from constants import *
from text import Text

# Constants
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def tuple_to_rectangle(tup):
    if len(tup) != 4:
        raise AttributeError("Incorrect input for tuple_to_rectangle. Input should be (x,y,w,h)")

    left = int(tup[0])
    top = int(tup[1])
    right = int(tup[0] + tup[2])
    bottom = int(tup[1] + tup[3])
    return dlib.rectangle(left=left, top=top, right=right, bottom=bottom)

def get_line_angle(line):
    """
    :param line: (start_x, start_y), (end_x, end_y)
    :return: direction of the line in radian.
    """
    (start_x, start_y), (end_x, end_y) = line
    return np.arctan2(float(end_y-start_y), float(end_x-start_x))


def is_line_horizontal(line, allowed_error=np.pi/18):
    """

    :param line: (start_x, start_y), (end_x, end_y)
    :return: True if line is nearly horizontal within +- allowed_error
    """
    angle = get_line_angle(line)
    if abs(angle - np.pi) < allowed_error or abs(angle) < allowed_error or abs(angle + np.pi) < allowed_error:
        return True
    else:
        return False

def is_line_vertical(line, allowed_error=np.pi/18):
    """

    :param line: (start_x, start_y), (end_x, end_y)
    :return: True if line is nearly horizontal within +- allowed_error
    """
    angle = get_line_angle(line)
    if abs(angle - np.pi/2) < allowed_error or abs(angle + np.pi/2) < allowed_error:
        return True
    else:
        return False

def get_image_from_surface(surface):
    pil_string_image = pygame.image.tostring(surface, "RGB", False)
    surface_width, surface_height = surface.get_rect().w, surface.get_rect().h
    image = np.asarray(Image.frombytes("RGB", (surface_width, surface_height), pil_string_image))
    return image

class FacialLandmarkDetector(object):
    _CALIBRATE_ROUNDS = 10
    _TEXT_COLOR = WHITE
    _CALIBRATE_BG_COLOR = BLACK
    _CALIBRATE_TEXT_TOP = (GAME_SCREEN_WIDTH / 2,  BATTLE_SCREEN_HEIGHT + (GAME_SCREEN_HEIGHT - BATTLE_SCREEN_HEIGHT) / 4)
    _CALIBRATE_TEXT_CENTER = (GAME_SCREEN_WIDTH / 2, (BATTLE_SCREEN_HEIGHT + GAME_SCREEN_HEIGHT) / 2)

    def __init__(self, camera_width, camera_height, name="PLAYER", path="shape_predictor_68_face_landmarks.dat"):
        # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
        if not os.path.isfile(path):
            raise IOError("Cannot find the facial landmark data file. Please download it from "
                          "http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks."
                          "dat.bz2 and extract that directly to the root folder of this project.")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path)
        self.camera_width = camera_width
        self.camera_height = camera_height
        # self.facial_landmark_predictor_width = facial_landmark_predictor_width
        # self.resize_ratio = (float(self.camera_width) / self.facial_landmark_predictor_width)
        # Calibration parameters
        self.calibrated = False
        self.calibrate_round_left = self._CALIBRATE_ROUNDS
        # All attributes starting with _cal is used internally for calibration purposes. They record the measured
        # values over the calibration rounds.
        self._cal_mouth_left_corner_to_center_dists = []
        self._cal_mouth_right_corner_to_center_dists = []
        self._cal_mouth = []
        self._cal_rolls = []
        self._cal_pitches = []
        self._cal_yaws = []
        # All attributes starting with norm is the calibrated value of a facial feature under normal conditions.
        self.norm_mouth_left_corner_to_center_dist = 0
        self.norm_mouth_right_corner_to_center_dist = 0
        self.norm_mouth = 0
        self.norm_roll = 0
        self.norm_pitch = 0
        self.norm_yaw = 0
        # Initialize text display for camera calibration
        self.text = Text()
        self.name = name




    def get_features(self, surface):
        image = get_image_from_surface(surface)
        # # The image needs to be resized to speed things up.
        # if self.camera_width != self.facial_landmark_predictor_width:
        #     image = imutils.resize(image, width=self.facial_landmark_predictor_width)

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
            # shape = shape * self.resize_ratio
            facial_features.append(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # (x, y, w, h) = (x * self.resize_ratio, y * self.resize_ratio, w * self.resize_ratio, h * self.resize_ratio)
            face_coordinates.append((x, y, w, h))
        return face_coordinates, facial_features

    def get_largest_face_index(self, face_coordinates):
        """

        :param face_coordinates: a list of (x, y, w, h), comes from calling get_features()
        :return: the index of the largest face.
        """
        if len(face_coordinates) == 0:
            raise AttributeError("You must feed in at least one set of face coordinates to get_largest_face_index().")
        ret = -1
        max_face_area = 0
        for i in range(len(face_coordinates)):
            face_area = abs(face_coordinates[i][2] - face_coordinates[i][0]) * abs(face_coordinates[i][3] - face_coordinates[i][1])
            if face_area > max_face_area:
                ret = i
                max_face_area = face_area
        if ret < 0:
            raise AttributeError("face coordinate areas should be larger than 0! No such face area found in %s"
                                 %(str(face_coordinates)))
        return ret

    def get_pose_diff(self, rvec):
        """

        :param rvec:
        :return: The difference in pose when comparing to the calibrated pose, i.e. yaw, pitch, and roll.
        """
        if not self.calibrated:
            raise AssertionError("Please use this function only after the facial landmark detector is calibrated.")
        # This is the difference in pose, i.e. yaw, pitch, and roll
        pose_diff = np.array([[self.norm_roll],
                             [self.norm_pitch],
                             [self.norm_yaw]]) - rvec
        return pose_diff

    def clear_calibrate_results(self):
        self.calibrated = False
        self.calibrate_round_left = self._CALIBRATE_ROUNDS
        # All attributes starting with _cal is used internally for calibration purposes. They record the measured
        # values over the calibration rounds.
        self._cal_mouth_left_corner_to_center_dists = []
        self._cal_mouth_right_corner_to_center_dists = []
        self._cal_mouth = []
        self._cal_rolls = []
        self._cal_pitches = []
        self._cal_yaws = []
        # All attributes starting with norm is the calibrated value of a facial feature under normal conditions.
        self.norm_mouth_left_corner_to_center_dist = 0
        self.norm_mouth_right_corner_to_center_dist = 0
        self.norm_mouth = 0
        self.norm_roll = 0
        self.norm_pitch = 0
        self.norm_yaw = 0


    def calibrate_face(self, camera_shot, head_pose_estimator, display):
        """
        This is a wrapper around the calibrate_face_aux, which does most of the heavy-lifting. This function makes sure
        the display is updated during calibration.
        """
        display.fill(self._CALIBRATE_BG_COLOR)
        self.text.blit_text_centered_at("Facial feature calibration for %s. " %(self.name),
                                            self._CALIBRATE_TEXT_TOP, display)
        ret = self.calibrate_face_aux(camera_shot, head_pose_estimator, display)
        pygame.display.flip()
        return ret

    def calibrate_face_aux(self, camera_shot, head_pose_estimator, display):
        """
        This function should be called before using other facial landmark utility functions. It measures the human face
        features under normal condition. In principle the pygame function should loop until the calibrate_face function
        returns true, then start the game.
        :param camera_shot: a pygame image. It should be the pygame camera.
        :param head_pose_estimator: a HeadPoseEstimator object.
        :return: True if calibration finishes. False if otherwise.
        """
        # Get the facial features of the largest face
        face_coordinates_list, facial_features_list = self.get_features(camera_shot)

        # Now show the image on the display.
        camera_shot_display = pygame.transform.scale(camera_shot, BATTLE_SCREEN_SIZE)
        display.blit(camera_shot_display, (0,0)) # Upper left corner. Take over the whole battle screen.

        if len(face_coordinates_list) == 0:
            self.text.blit_text_centered_at("Can't see any face. Try to move around a little.",
                                            self._CALIBRATE_TEXT_CENTER, display)
            self.clear_calibrate_results()
            return False

        face_index = self.get_largest_face_index(face_coordinates_list)
        head_pose = head_pose_estimator.head_pose_estimation(facial_features_list[face_index])
        assert head_pose is not None
        rvec, tvec, axes = head_pose

        # Check if the head is straight. That is, check whether z axis is vertical.
        if not (is_line_vertical(axes[2])):
            self.text.blit_text_centered_at("Still calibrating... Please make sure your head is straight.",
                                            self._CALIBRATE_TEXT_CENTER, display)
            self.clear_calibrate_results()
            return False

        # Display 2d facial features, if there is any.
        # xyz axes.
        axes = np.array(axes)
        # Resized to the current battle display width and height
        axes_x_resized = axes[:,:,0]  / (float(CAMERA_INPUT_WIDTH) / BATTLE_SCREEN_WIDTH)
        axes_y_resized = axes[:,:,1]  / (float(CAMERA_INPUT_HEIGHT) / BATTLE_SCREEN_HEIGHT)
        axes = np.stack((axes_x_resized, axes_y_resized), axis=2)

        pygame.draw.line(display, RED, axes[0,0], axes[0,1])
        pygame.draw.line(display, GREEN, axes[1,0], axes[1,1])
        pygame.draw.line(display, BLUE, axes[2,0], axes[2,1])
        # Draw facial features...
        for feature in facial_features_list[face_index]:
            # circle(Surface, color, pos, radius, width=0)
            feature = (int(feature[0] / (float(CAMERA_INPUT_WIDTH) / BATTLE_SCREEN_WIDTH)),
                       int(feature[1] / (float(CAMERA_INPUT_HEIGHT) / BATTLE_SCREEN_HEIGHT)))
            pygame.draw.circle(display, WHITE, feature, 2)

        # Now calculate the rotational invariant facial features

        facial_features = head_pose_estimator.facial_features_to_3d(facial_features_list[face_index], rvec, tvec)

        self._cal_rolls.append(rvec[0])
        self._cal_pitches.append(rvec[1])
        self._cal_yaws.append(rvec[2])


        self._cal_mouth_left_corner_to_center_dists.append(get_mouth_left_corner_to_center_dist(facial_features))
        self._cal_mouth_right_corner_to_center_dists.append(get_mouth_right_corner_to_center_dist(facial_features))
        self._cal_mouth.append(get_mouth_right_to_left_dist(facial_features))


        self.calibrate_round_left -= 1
        if self.calibrate_round_left <= 0:
            self.calibrated = True
            self.norm_mouth_left_corner_to_center_dist = np.average(self._cal_mouth_left_corner_to_center_dists)
            self.norm_mouth_right_corner_to_center_dist = np.average(self._cal_mouth_right_corner_to_center_dists)
            self.norm_mouth = np.average(self._cal_mouth)
            self.norm_roll = np.average(self._cal_rolls)
            self.norm_pitch = np.average(self._cal_pitches)
            self.norm_yaw = np.average(self._cal_yaws)
            if DEBUG_LEVEL >= DEBUG_PRINT_ALL:
                print("Calibration finishes! mouth_left_corner: %f, mouth_right_corner: %f, norm_roll: %f, norm_pitch: %f"
                      ", norm_yaw: %f." %(self.norm_mouth_left_corner_to_center_dist,
                                          self.norm_mouth_right_corner_to_center_dist,
                                          self.norm_roll, self.norm_pitch, self.norm_yaw))
            return True
        else:
            self.text.blit_text_centered_at("Keep still! Need %d more images to finish calibration."
                                            %(self.calibrate_round_left),
                                            self._CALIBRATE_TEXT_CENTER, display)
            return False

class HeadPoseEstimator():
    # Adapted from https://github.com/mpatacchiola/deepgaze/blob/master/examples/ex_pnp_head_pose_estimation_webcam.py
    # Other useful resources: http://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/#code

    # From https://github.com/chili-epfl/attention-tracker/blob/master/src/head_pose_estimation.hpp
    P3D_SELLION = [0., 0., 0.]  # 27
    P3D_RIGHT_EYE = [-20., -65.5, -5.]  # 36
    P3D_LEFT_EYE = [-20., 65.5, -5.]  # 45
    P3D_RIGHT_EAR = [-100., -77.5, -6.]  # 0
    P3D_LEFT_EAR = [-100., 77.5, -6.]  # 16
    P3D_NOSE = [21.0, 0., -48.0]  # 30
    P3D_STOMMION = [10.0, 0., -75.0]
    P3D_MENTON = [0., 0., -133.0]  # 8

    TRACKED_POINTS = [27, 36, 45, 0, 16, 8, 30, ]

    landmarks_3D = np.float32([P3D_SELLION, P3D_RIGHT_EYE, P3D_LEFT_EYE, P3D_RIGHT_EAR, P3D_LEFT_EAR, P3D_MENTON, P3D_NOSE])

    def __init__(self, camera_w, camera_h):
        # TODO: when camera width and hight is not 640 and 480, it doesn't work for some reason...
        # Defining the camera matrix.
        # To have better result it is necessary to find the focal
        # lenght of the camera. fx/fy are the focal lengths (in pixels)
        # and cx/cy are the optical centres. These values can be obtained
        # roughly by approximation, for example in a 640x480 camera:
        # cx = 640/2 = 320
        # cy = 480/2 = 240
        # fx = fy = cx/tan(60/2 * pi / 180) = 554.26
        c_x = camera_w / 2
        c_y = camera_h / 2
        f_x = c_x / np.tan(60 / 2 * np.pi / 180)
        f_y = f_x

        # Estimated camera matrix values.
        self.camera_matrix = np.float32([[f_x, 0.0, c_x],
                                         [0.0, f_y, c_y],
                                         [0.0, 0.0, 1.0]])

        print("Estimated camera matrix: \n" + str(self.camera_matrix) + "\n")

        # Commented out for now.  See this for more detail about how to get calibrated camera matrix and
        # camera distortion:
        # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
        self.camera_distortion = np.float32([0, 0, 0, 0, 0])

    def head_pose_estimation(self, facial_features):
        """
        TODO: this version also returns a list of 3d facial features
        :param facial_features: format: numpy array of shape (68, 2)
        :return: Rotation vector, translation vector, and the three axis (lines) projected in the image plane.
        """
        landmarks_2D = np.array(facial_features)[self.TRACKED_POINTS].astype(np.float32)
        # Applying the PnP solver to find the 3D pose
        # of the head from the 2D position of the
        # landmarks.
        # retval - bool
        # rvec - Output rotation vector that, together with tvec, brings
        # points from the model coordinate system to the camera coordinate system.
        # It corresponds to roll, pitch, and yaw.
        # Yaw is looking left or right. -pi ~ pi
        # Pitch is looking up and down. 0~ pi
        # Roll is tilting face to left or right. -pi ~ pi
        # tvec - Output translation vector.
        retval, rvec, tvec = cv2.solvePnP(self.landmarks_3D,
                                          landmarks_2D,
                                          self.camera_matrix, self.camera_distortion)
        if retval:
            # Now we project the 3D points into the image plane
            # Creating a 3-axis to be used as reference in the image.
            axis = np.float32([[200, 0, 0],
                               [0, 200, 0],
                               [0, 0, 200]])


            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, self.camera_matrix, self.camera_distortion)
            imgpts = imgpts[:, 0, :]  # Format: numpy array with shape (3,2), basically a list of (x,y)

            # Instead of returning the points themselves, return the line from Sellion to those points, which will be
            # more useful.
            lines = [(facial_features[SELLION_INDEX], imgpts[0]),
                     (facial_features[SELLION_INDEX], imgpts[1]),
                     (facial_features[SELLION_INDEX], imgpts[2])]

            return rvec, tvec, lines
        else:

            if DEBUG_LEVEL >= DEBUG_PRINT_UNEXPECTED_ERROR:
                print("Warning: failed to solve pnp in head pose estimation. Returning None")
            return None

    def two_d_to_three_d(self, points, rvec, tvec, const_x=1, precalculated_s = None):
        # http://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
        # It looks like we need some extra parameters to decide the 3d coordinate of the face. I need some invariant,
        # some reference so that I can decide 3d points from 2d. Or I can assume that the x coordinate does not change
        # and compute y and z. Then every score I only use proportion. But is that invariant to rotation? I think so
        # because the rotation matrix forces the reconstructed 3d points to be facing forward. Yeah. That should work.

        # No that actually does not work, because I need the correct distance to treat x and y on the same scale.
        # Otherwise if I assume depth is 1 and it's actually 100, then when I turn my head, the depth change is much
        # smaller than the actual change and the distance is measured incorrectly.
        # Well shit. How am I supposed to know the distance? Should I just stick some sort of calibration
        # piece of paper on them?? Or I can assume that the distance between point 0 and point 17 is some
        # value and calculate based on that. That will induce some error but it should work better than what I
        # have now.
        # The mouth opening works because the points are close enough to each other. So blinking should
        # also work. Mouth corner is hard with rotation.


        is_ndarray = False
        if isinstance(points, np.ndarray):
            points = [points]
            is_ndarray = True
        else:
            assert isinstance(points, list)

        # First turn rvec into rotation matrix
        r_matrix, jacobian = cv2.Rodrigues(rvec)
        r_inv = np.linalg.inv(r_matrix)
        m_inv = np.linalg.inv(self.camera_matrix)
        r_inv_m_inv = np.dot(r_inv, m_inv)
        r_inv_t = np.dot(r_inv, tvec)

        if not is_ndarray and precalculated_s is None:
            # When the input is a list of points, we can no longer assume that they are all on the same plane
            # parallel to the camera. We choose point 0 and 16 as the two reference points and
            # all transformation will be calculated with respect to that point alone.
            r_inv_m_inv_left_ref = np.dot(r_inv_m_inv,
                                                 np.expand_dims(np.concatenate((points[0], [1])), axis=1))
            r_inv_m_inv_right_ref = np.dot(r_inv_m_inv,
                                                 np.expand_dims(np.concatenate((points[16], [1])), axis=1))
        ret=[]

        for i, point in enumerate(points):
            # Solve for s using the const_x condition.
            uv1 = np.expand_dims(np.concatenate((point, [1])), axis=1)  # Shape 3x1
            r_inv_m_inv_uv1 = np.dot(r_inv_m_inv, uv1)
            if precalculated_s is None:
                if is_ndarray:
                    # No need to use a reference point.
                    s = (const_x + r_inv_t[0, 0]) / (r_inv_m_inv_uv1[0, 0])
                else:
                    # Calculate the s based on the left and right side of the face.
                    s = (self.P3D_LEFT_SIDE[1] - self.P3D_RIGHT_SIDE[1]) / \
                        dist.euclidean(r_inv_m_inv_left_ref, r_inv_m_inv_right_ref)
                    if i == 0:
                        if DEBUG_LEVEL >= DEBUG_PRINT_ALL:
                            print("Scale factor: %f" %s)
            else:
                s = precalculated_s
            three_d_point = r_inv_m_inv_uv1 * s - r_inv_t
            assert three_d_point.shape == (3,1)

            three_d_point = three_d_point[:,0]
            ret.append(three_d_point)

        if is_ndarray:
            # If input is only one point, then return one point
            return ret[0]
        else:
            # If the input is a list of points, then the return is also a list of points.
            return ret

    def facial_features_to_3d(self, facial_features, rvec, tvec, const_x=1):
        scale = self.calc_scale_factor(np.array(facial_features)[self.TRACKED_POINTS].tolist(), rvec, tvec)
        original_shape = facial_features.shape
        ret = self.two_d_to_three_d(facial_features.tolist(), rvec, tvec, const_x, precalculated_s=scale)
        ret = np.array(ret)
        assert ret.shape == (original_shape[0], original_shape[1]+1)
        return ret

    def calc_scale_factor(self, landmark_2d, rvec, tvec):
        assert len(landmark_2d) == len(self.landmarks_3D)

        r_matrix, jacobian = cv2.Rodrigues(rvec)
        transition_matrix = np.concatenate((r_matrix, tvec), axis=1)
        s_list = []
        for i, landmark in enumerate(landmark_2d):
            rhs = np.dot(np.dot(self.camera_matrix, transition_matrix), np.expand_dims(np.concatenate((self.landmarks_3D[i], [1])), axis=1))
            uv1 = np.expand_dims(np.concatenate((landmark, [1])), axis=1)

            s = np.linalg.lstsq(rhs, uv1)[0]
            s = s[0,0]
            s_list.append(s)

        ret = np.average(s_list)
        return 1 / ret

# NOTE: All facial feature indices should be the number on the facial feature picture - 1, because index start at 0.

def get_mouth_open_score(facial_features_3d):
    """

    :param facial_features_3d: format: numpy array of shape (68, 3)
    :return: The degree of mouth opening
    """
    assert isinstance(facial_features_3d, np.ndarray)
    # Get point 62, 64, 66, 68. Don't forget their index is one minus that.

    # The width will depend on x and y, but not z.

    w = float(dist.euclidean(facial_features_3d[61, :2], facial_features_3d[63, :2]) + dist.euclidean(facial_features_3d[65, :2], facial_features_3d[67, :2])) / 2.0
    # The height will only be dependent on z.
    h = float(abs(facial_features_3d[61, 2] - facial_features_3d[67, 2]) + abs(facial_features_3d[63, 2] - facial_features_3d[65, 2])) / 2.0
    # print("W: %.1f H: %.1f" %(w,h))
    if w == 0:
        if DEBUG_LEVEL >= DEBUG_PRINT_UNEXPECTED_ERROR:
            print("Wierd. Width of the mouth should not normally be 0.")
        return 0
    else:
        return h / w


def get_mouth_left_corner_to_center_dist(facial_features_3d):
    assert isinstance(facial_features_3d, np.ndarray)
    # Can't use center of mouth because the facial landmark detector distorts the mouth when I turn to the side.
    nose_to_menton_point = np.average((facial_features_3d[33], facial_features_3d[8]), axis=0)
    left_mouth_point = facial_features_3d[60]
    between_eyes_to_menton_point = np.average((facial_features_3d[SELLION_INDEX], facial_features_3d[8]), axis=0)
    # Ignore z the height
    ret = np.average((dist.euclidean(nose_to_menton_point[:2], left_mouth_point[:2]), dist.euclidean(between_eyes_to_menton_point[:2], left_mouth_point[:2])))
    return ret

def get_mouth_left_corner_score(facial_features_3d, normal_d):
    """
    :param facial_features_3d: format: numpy array of shape (68, 2)
    :param normal_d: The distance of the mouth left corner to center under normal conditions.
    :return: A ratio between (distance from the left mouth corner to the right face edge) and the face width.
    Larger score means closer to boarder of the face - thus more distorted facial expression.
    """
    assert isinstance(facial_features_3d, np.ndarray)
    d = get_mouth_left_corner_to_center_dist(facial_features_3d)
    # between_eyes_to_menton_distance = dist.euclidean(facial_features_3d[33], facial_features_3d[8])
    return d / normal_d


def get_mouth_right_corner_to_center_dist(facial_features_3d):
    assert isinstance(facial_features_3d, np.ndarray)
    # Can't use center of mouth because the facial landmark detector distorts the mouth when I turn to the side.
    nose_to_menton_point = np.average((facial_features_3d[33], facial_features_3d[8]), axis=0)
    right_mouth_point = facial_features_3d[54]
    between_eyes_to_menton_point = np.average((facial_features_3d[SELLION_INDEX], facial_features_3d[8]), axis=0)
    # Ignore z the height
    ret = np.average((dist.euclidean(nose_to_menton_point[:2], right_mouth_point[:2]),
                      dist.euclidean(between_eyes_to_menton_point[:2], right_mouth_point[:2])))
    return ret

def get_mouth_right_corner_score(facial_features_3d, normal_d):
    """
    :param facial_features_3d: format: numpy array of shape (68, 2)
    :param normal_d: The distance of the mouth the right corner to center under normal conditions.
    :return: A ratio between the current distance from the right mouth corner to the center and the pre-measured
    distance.
    Larger score means closer to boarder of the face - thus more distorted facial expression.
    """
    assert isinstance(facial_features_3d, np.ndarray)
    d = get_mouth_left_corner_to_center_dist(facial_features_3d)
    # Then compare that with a pre-measured right mouth corner to center of mouth
    return d / normal_d

def get_mouth_right_to_left_dist(facial_features_3d):
    assert isinstance(facial_features_3d, np.ndarray)
    # hacky but works
    return max(np.std(np.concatenate((facial_features_3d[48:51, 1], facial_features_3d[57:62, 1], facial_features_3d[66:67, 1], ))),
               np.std(np.concatenate((facial_features_3d[51:57, 1], facial_features_3d[62:66, 1],))), )

def get_mouth_right_to_left_score(facial_features_3d, normal_d):
    d = get_mouth_right_to_left_dist(facial_features_3d)
    return d / normal_d

def eye_aspect_ratio(eye):
    # Adapted from http://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# Note: I noticed that one I close one of my eyes, both left and right score decreases, probably because that's how the
# muscles work.
def get_left_blink_score(facial_features_3d):
    left_eye = facial_features_3d[lStart:lEnd]
    left_e_a_r = eye_aspect_ratio(left_eye)
    return left_e_a_r

def get_right_blink_score(facial_features_3d):
    right_eye = facial_features_3d[rStart:rEnd]
    right_e_a_r = eye_aspect_ratio(right_eye)
    return right_e_a_r


def get_any_eye_blink_ear(facial_features_3d):
    left_e_a_r = get_left_blink_score(facial_features_3d)
    right_e_a_r = get_right_blink_score(facial_features_3d)
    # Take the smallest one.
    ear = min(left_e_a_r, right_e_a_r)
    return ear


def get_any_eye_blink_rounds(facial_features_3d, num_rounds, cal_eye, threshold=EYE_AR_THRESH):
    """

    :param facial_features_3d:
    :param num_rounds: Number of rounds that any of the eyes have ear level lower than the threshold.
    :param threshold: The eyes are treated as closed if the 'ear' level is below this threshold.
    :return: new num_rounds = 0 if eyes are open, += 1 if eyes are closed.
    """
    if cal_eye <= 0:
        raise ValueError("The calibrated eye EAR value is not positive!")
    ear = get_any_eye_blink_ear(facial_features_3d)
    if ear / cal_eye <= threshold:
        return num_rounds + 1
    else:
        return 0

def get_direction_from_line(line):
    """
    This is for controlling the movement of cat or deafy using head pose.
    :param pose_diff: The difference in pose when comparing to the calibrated pose. should be (yaw, pitch, roll)
    :return: (bool - true if the object should move ,a float from 0~2pi representing the direction)
    """

    dy = line[0][1] - line[1][1]
    # In pygame display left and right is reversed from what we usually perceive.
    dx = -(line[0][0] - line[1][0])

    if dist.euclidean(line[0], line[1]) <= POSE_MOVE_LOWER_THRESHOLD:
        return (False, 0)
    else:
        direction = np.arctan2(dy, dx) + math.pi  # [0, 2pi]
        return (True, direction)

def get_take_photo_score(mouth_open_score, direction_line, facial_features_3d):
    """
    Gets a score for automatically taking photos of funny looking expressions. The higher the score the better.
    :param mouth_open_score:
    :param blink_ear:
    :param direction_line:
    :param facial_features_3d:
    :return:
    """
    blink_ear = get_any_eye_blink_ear(facial_features_3d)
    ret = mouth_open_score * (0.1 / (0.1 + blink_ear)) * dist.euclidean(direction_line[0], direction_line[1])
    return ret


# Face swapping https://matthewearl.github.io/2015/07/28/switching-eds-with-python/