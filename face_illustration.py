import numpy as np
import pygame
import pygame.camera
from constants import *


class Face(pygame.sprite.Sprite):
    """ This class represents the illustration of the facial landmarks.  It can be used for tutorial or for real-time
    user feedback."""
    _WIDTH = FACE_ILLUS_WIDTH
    _HEIGHT = FACE_ILLUS_HEIGHT
    _BG_COLOR = BLACK
    _BUFFER_SIZE = 30
    _WINDOW = (_WIDTH, _HEIGHT)
    _SELLION_CENTER_POS = (_WIDTH / 2, _HEIGHT / 4)

    def __init__(self, camera_window, pos=GAME_SCREEN_RECT.bottomleft):
        """

        :param camera_window: (width, height)
        :param pos:
        """
        pygame.sprite.Sprite.__init__(self, self.containers)

        self.pos = pos
        self.image = pygame.Surface([self._WIDTH, self._HEIGHT])
        self.image.fill(self._BG_COLOR)

        self.rect = self.image.get_rect(bottomleft=pos)
        self.centered_facial_features_2d = None
        self.centered_projection_lines = None
        self.face_coordinates = None


        self.camera_window = camera_window

        # The boundary here doesn't seem to be correct for some reason.
        self.move_threshold_ellipse_loc = Rect(0, 0,
                                               POSE_MOVE_LOWER_THRESHOLD * 2 * self._WINDOW[0] / self.camera_window[0],
                                               POSE_MOVE_LOWER_THRESHOLD * 2 * self._WINDOW[1] / self.camera_window[1])
        self.move_threshold_ellipse_loc.center = self._SELLION_CENTER_POS

    def update(self):
        # First clear image
        self.image.fill(self._BG_COLOR)

        if self.centered_facial_features_2d is None or self.centered_projection_lines is None \
                or self.face_coordinates is None:
            # Async process has not yet filled those parameters,
            pass
            # raise AttributeError("You must first feed in the facial features using refresh_features().")
        else:
            pygame.draw.ellipse(self.image, RED, self.move_threshold_ellipse_loc, 1)  # TODO: change color when moving.

            # Line that controls moving
            pygame.draw.line(self.image, WHITE, self.centered_projection_lines[0,0],
                             self.centered_projection_lines[0,1])
            # Draw facial features...
            for feature in self.centered_facial_features_2d:
                # circle(Surface, color, pos, radius, width=0)
                pygame.draw.circle(self.image, (0, 0, 255), feature, 2)

    def refresh_features(self, face_coordinates, facial_features_2d, projection_lines):
        self.face_coordinates = face_coordinates
        self.centered_facial_features_2d = np.array([self.resize_point(feature) for feature in facial_features_2d])
        # Recenter facial features so that the sellion is in the middle of the window.
        self.centered_projection_lines = np.array([[self.resize_point(line[0]),self.resize_point(line[1])] for line in projection_lines]) \
                                         - self.centered_facial_features_2d[SELLION_INDEX] \
                                         + np.array(self._SELLION_CENTER_POS)
        self.centered_facial_features_2d = self.centered_facial_features_2d \
                                           - self.centered_facial_features_2d[SELLION_INDEX] \
                                           + np.array(self._SELLION_CENTER_POS)
        self.centered_facial_features_2d = self.centered_facial_features_2d.astype(np.int)

    def resize_point(self, point):
        """

        :param point: (x,y) under camera window size.
        :return: The point under the current window size, resized with respect to the center of the window.
        """
        return (int((float(point[0] - self.face_coordinates[0]) / self.face_coordinates[2] * self._WINDOW[0])),
                int((float(point[1] - self.face_coordinates[1]) / self.face_coordinates[3] * self._WINDOW[1])))