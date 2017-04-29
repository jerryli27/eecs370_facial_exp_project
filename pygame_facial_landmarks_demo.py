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

import pygame
import pygame.camera
from pygame.locals import *

from facial_landmark_util import get_mouth_open_degree

#game constants
SCREEN_HEIGHT=480
SCREEN_WIDTH=640
SCREEN_RECT= Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
FACIAL_LANDMARK_PREDICTOR_WIDTH = 320
RESIZE_RATIO = (float(SCREEN_WIDTH) / FACIAL_LANDMARK_PREDICTOR_WIDTH)
MOUTH_RATIO_LOWER_THRESHOLD = 0.2

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
        self.camera = pygame.camera.Camera(cam_id, self.size, "RGB")

        # starts the camera
        self.camera.start()

        self.clock = pygame.time.Clock()

        # create a surface to capture to.  for performance purposes, you want the
        # bit depth to be the same as that of the display surface.
        self.snapshot = pygame.surface.Surface(self.size, 0, self.display)

    def get_camera_snapshot(self):
        # if you don't want to tie the framerate to the camera, you can check and
        # see if the camera has an image ready.  note that while this works
        # on most cameras, some will never return true.
        if 0 and self.camera.query_image():
            # capture an image
            self.snapshot = self.camera.get_image(self.snapshot)
        if 0:
            self.snapshot = self.camera.get_image(self.snapshot)
            # blit it to the display surface.  simple!
            self.display.blit(self.snapshot, (0,0))
        else:
            self.snapshot = self.camera.get_image(self.display)

    def main(self, facial_landmark_detector,):
        going = True
        while going:
            events = pygame.event.get()
            for e in events:
                if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                    going = False
                if e.type == KEYDOWN:
                    if e.key in range(K_0, K_0+10) :
                        self.init_cams(e.key - K_0)

            self.get_camera_snapshot()
            # Now use the facial landmark defector
            # This step decreases the frame rate from 30 fps to 6fps. So we need to do something about it.
            # The speed is directly related to FACIAL_LANDMARK_PREDICTOR_WIDTH.
            face_coordinates, facial_features = facial_landmark_detector.get_features(self.snapshot)
            if len(face_coordinates) > 0:
                # For now simply get the first face and update the sprites.
                pygame.draw.rect(self.display, (0,255,0), face_coordinates[0], 2)
                assert facial_features[0].shape[0] == 68
                for i in range(68):
                    self.circles[i].move(facial_features[0][i])

                print("Detected %d face%s." % (len(face_coordinates),
                                               "s" if len(face_coordinates) > 1 else ""))
                # Assume the first face is the target for now.
                mouth_open_degree = get_mouth_open_degree(facial_features[0])
                if mouth_open_degree >= MOUTH_RATIO_LOWER_THRESHOLD:
                    print("Mouth open degree: %f" % (mouth_open_degree))
                else:
                    print("Mouth closed degree: %f" % (mouth_open_degree))

            # clear/erase the last drawn sprites
                self.all.clear(self.display, self.background)

            # update the facial feature sprites only if at least one face is detected.
            if len(face_coordinates) > 0:
                self.facial_feature_group.update()
                # draw the scene
                dirty = self.facial_feature_group.draw(self.display)
                pygame.display.update(dirty)
            pygame.display.flip()
            self.clock.tick()
            print (self.clock.get_fps())

class FacialLandmarkDetector(object):
    def __init__(self, path="shape_predictor_68_face_landmarks.dat"):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path)

    def get_features(self, surface):
        pil_string_image = pygame.image.tostring(surface, "RGBA", False)
        image = np.asarray(Image.frombytes("RGBA", (SCREEN_WIDTH, SCREEN_HEIGHT), pil_string_image))
        # The image needs to be resized to speed things up.
        if SCREEN_WIDTH != FACIAL_LANDMARK_PREDICTOR_WIDTH:
            image = imutils.resize(image, width=FACIAL_LANDMARK_PREDICTOR_WIDTH)
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
            shape = shape * RESIZE_RATIO
            facial_features.append(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            (x, y, w, h) = (x * RESIZE_RATIO, y * RESIZE_RATIO, w * RESIZE_RATIO, h * RESIZE_RATIO)
            face_coordinates.append((x, y, w, h))

        return face_coordinates, facial_features

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

    VideoCapturePlayer().main(FacialLandmarkDetector())
    pygame.quit()

if __name__ == '__main__':
    main()
