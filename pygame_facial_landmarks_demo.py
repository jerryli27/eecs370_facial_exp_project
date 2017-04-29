#!/usr/bin/env python

# It's an adaptation of the camera tutorial in pygame that does facial landmark detection. It can be incorporated into
# other parts of the game in the future.


from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
from PIL import Image
import scipy.misc

import pygame
import pygame.camera
from pygame.locals import *

from facial_landmark_util import get_mouth_open_score, get_mouth_left_corner_score, FacialLandmarkDetector, tuple_to_rectangle
from align_dlib import AlignDlib

#game constants
SCREEN_HEIGHT=480
SCREEN_WIDTH=640
SCREEN_RECT= Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
FACIAL_LANDMARK_PREDICTOR_WIDTH = 320
RESIZE_RATIO = (float(SCREEN_WIDTH) / FACIAL_LANDMARK_PREDICTOR_WIDTH)
MOUTH_RATIO_LOWER_THRESHOLD = 0.2
MOUTH_LEFT_CORNER_THRESHOLD = 0.6

CAMERA_INPUT_HEIGHT = 480
CAMERA_INPUT_WIDTH = 640
CAMERA_DISPLAY_HEIGHT = SCREEN_HEIGHT
CAMERA_DISPLAY_WIDTH = SCREEN_WIDTH
CAMERA_DISPLAY_SIZE = (CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT)


main_dir = os.path.split(os.path.abspath(__file__))[0]


def load_image(file):
    "loads an image, prepares it for play"
    file = os.path.join(main_dir, 'data', file)
    try:
        surface = pygame.image.load(file)
    except pygame.error:
        raise SystemExit('Could not load image "%s" %s'%(file, pygame.get_error()))
    return surface.convert()

def load_images(*files):
    imgs = []
    for file in files:
        imgs.append(load_image(file))
    return imgs


class dummysound:
    def play(self): pass

def load_sound(file):
    if not pygame.mixer: return dummysound()
    file = os.path.join(main_dir, 'data', file)
    try:
        sound = pygame.mixer.Sound(file)
        return sound
    except pygame.error:
        print ('Warning, unable to load, %s' % file)
    return dummysound()

class VideoCapturePlayer(object):

    size = ( SCREEN_WIDTH, SCREEN_HEIGHT )
    def __init__(self, **argd):
        self.__dict__.update(**argd)
        super(VideoCapturePlayer, self).__init__(**argd)

        # create a display surface. standard pygame stuff
        self.display = pygame.display.set_mode( self.size, 0 )
        self.init_cams(0)
        self.background = pygame.Surface(SCREEN_RECT.size)

        Circle.images = [load_image('circle.png')]
        # Initialize Game Groups
        self.all = pygame.sprite.RenderUpdates()
        self.facial_feature_group = pygame.sprite.RenderUpdates()

        # assign default groups to each sprite class
        Circle.containers = self.all, self.facial_feature_group

        self.circles = [Circle() for _ in range(68)]

    def init_cams(self, which_cam_idx):

        # gets a list of available cameras.
        self.clist = pygame.camera.list_cameras()
        print (self.clist)

        if not self.clist:
            raise ValueError("Sorry, no cameras detected.")

        try:
            cam_id = self.clist[which_cam_idx]
        except IndexError:
            cam_id = self.clist[0]

        # creates the camera of the specified size and in RGB colorspace
        self.camera = pygame.camera.Camera(cam_id, (CAMERA_INPUT_WIDTH, CAMERA_INPUT_HEIGHT), "RGB")

        # starts the camera
        self.camera.start()

        self.clock = pygame.time.Clock()

        # create a surface to capture to.  for performance purposes, you want the
        # bit depth to be the same as that of the display surface.
        self.camera_shot_raw = pygame.surface.Surface((CAMERA_INPUT_WIDTH, CAMERA_INPUT_HEIGHT), 0, self.display)
        self.camera_default_display_location = (SCREEN_WIDTH - CAMERA_DISPLAY_WIDTH, SCREEN_HEIGHT - CAMERA_DISPLAY_HEIGHT)

    def get_camera_shot(self):
        # For now, only get the camera shot and store it in self.camera_shot_raw.
        # if you don't want to tie the framerate to the camera, you can check and
        # see if the camera has an image ready.  note that while this works
        # on most cameras, some will never return true.
        # if 0 and self.camera.query_image():
        #     # capture an image
        #     self.camera_shot_raw = self.camera.get_image(self.camera_shot_raw)
        # if 0:
        #     self.camera_shot_raw = self.camera.get_image(self.camera_shot_raw)
        #     # blit it to the display surface.  simple!
        #     self.display.blit(self.camera_shot_raw, (0, 0))
        # else:
        #     self.camera_shot_raw = self.camera.get_image(self.display)
        self.camera_shot_raw = self.camera.get_image(self.camera_shot_raw)
        self.camera_shot = pygame.transform.scale(self.camera_shot_raw, CAMERA_DISPLAY_SIZE)

    def blit_camera_shot(self, blit_location):
        """

        :param blit_location: tuple with format (x, y)
        :return:
        """
        self.display.blit(self.camera_shot, blit_location)

    def main(self, facial_landmark_detector,):

        align_dlib_object = AlignDlib()

        going = True
        while going:
            events = pygame.event.get()
            for e in events:
                if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                    going = False
                if e.type == KEYDOWN:
                    if e.key in range(K_0, K_0+10) :
                        self.init_cams(e.key - K_0)

            self.get_camera_shot()
            # Now use the facial landmark defector
            # This step decreases the frame rate from 30 fps to 6fps. So we need to do something about it.
            # The speed is directly related to FACIAL_LANDMARK_PREDICTOR_WIDTH.
            face_coordinates, facial_features = facial_landmark_detector.get_features(self.camera_shot_raw)
            if len(face_coordinates) > 0:
                # For now simply get the first face and update the sprites.
                pygame.draw.rect(self.display, (0,255,0), face_coordinates[0], 2)
                assert facial_features[0].shape[0] == 68
                for i in range(68):
                    self.circles[i].move(facial_features[0][i])

                print("Detected %d face%s." % (len(face_coordinates),
                                               "s" if len(face_coordinates) > 1 else ""))
                # Assume the first face is the target for now.
                mouth_open_degree = get_mouth_open_score(facial_features[0])
                if mouth_open_degree >= MOUTH_RATIO_LOWER_THRESHOLD:
                    print("Mouth open degree: %f" % (mouth_open_degree))
                else:
                    print("Mouth closed degree: %f" % (mouth_open_degree))

                # mouth_left_corner_score = get_mouth_left_corner_score(facial_features[0])
                # if mouth_left_corner_score >= MOUTH_LEFT_CORNER_THRESHOLD:
                #     print("Mouth left corner score OK: %f" %(mouth_left_corner_score))
                # else:
                #     print("Mouth left corner score failed: %f" %(mouth_left_corner_score))

                # aligned_face = align_dlib_object.align(128, rgbImg=scipy.misc.imresize(
                #     facial_landmark_detector.get_image_from_surface(self.camera_shot_raw)[..., :3], (128, 128)),
                #                                        bb=tuple_to_rectangle(face_coordinates[0]),
                #                                        landmarks=facial_features[0])
                # aligned_face = align_dlib_object.align(128, rgbImg=
                #     facial_landmark_detector.get_image_from_surface(self.camera_shot_raw)[..., :3],
                #                                        bb=tuple_to_rectangle(face_coordinates[0]),
                #                                        landmarks=facial_features[0])
                aligned_face = align_dlib_object.align(128, rgbImg=imutils.resize(
                facial_landmark_detector.get_image_from_surface(self.camera_shot_raw)[..., :3],FACIAL_LANDMARK_PREDICTOR_WIDTH),)

            # clear/erase the last drawn sprites
            self.all.clear(self.display, self.background)
            self.blit_camera_shot((0,0))

            # update the facial feature sprites only if at least one face is detected.
            if len(face_coordinates) > 0:
                self.facial_feature_group.update()
                # draw the scene
                dirty = self.facial_feature_group.draw(self.display)
                pygame.display.update(dirty)
                if aligned_face is not None:
                    aligned_face_surface = pygame.surfarray.make_surface(aligned_face)
                    self.display.blit(aligned_face_surface, (0,0))
            pygame.display.flip()
            self.clock.tick()
            print (self.clock.get_fps())

# This is for testing a custom Sprite object. Usually if we want to draw a circle we can use pygame.draw.circle. You
# can load different images to the circle.
class Circle(pygame.sprite.Sprite):
    speed = -11
    images = []
    def __init__(self, pos=SCREEN_RECT.midbottom):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = self.images[0]
        self.rect = self.image.get_rect(midbottom=pos)

    def move(self, pos):
        self.rect= self.image.get_rect(midbottom=pos)
        self.rect = self.rect.clamp(SCREEN_RECT)

def main():
    pygame.init()
    pygame.camera.init()

    VideoCapturePlayer().main(FacialLandmarkDetector(SCREEN_WIDTH, SCREEN_HEIGHT, FACIAL_LANDMARK_PREDICTOR_WIDTH))
    pygame.quit()

if __name__ == '__main__':
    main()
