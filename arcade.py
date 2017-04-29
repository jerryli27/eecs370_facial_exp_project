#!/usr/bin/env python

# This file builds a simple 2d arcade game.

import argparse
import os
import random

import pygame
import pygame.camera
from pygame.locals import *

from facial_landmark_util import FacialLandmarkDetector, get_mouth_open_degree
from sprite_sheet import SpriteSheet


# Command line argument parser.
parser = argparse.ArgumentParser()

parser.add_argument("--no_camera", dest="camera", action="store_false",
                    help="Turn off camera and use keyboard control.")
parser.set_defaults(camera=True)

ARGS = parser.parse_args()

#game constants
SCREEN_HEIGHT=480
SCREEN_WIDTH=640
SCREEN_RECT= Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
FACIAL_LANDMARK_PREDICTOR_WIDTH = 320
CAMERA_INPUT_HEIGHT = 480
CAMERA_INPUT_WIDTH = 640
CAMERA_DISPLAY_HEIGHT = SCREEN_HEIGHT / 8
CAMERA_DISPLAY_WIDTH = SCREEN_WIDTH / 8
CAMERA_DISPLAY_SIZE = (CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT)
BACKGROUND_OBJECT_HEIGHT = 32
BACKGROUND_OBJECT_WIDTH = 32
DEAFY_SCREEN_POS = (SCREEN_WIDTH/8, SCREEN_HEIGHT*7/8)
MAX_FPS = 30
GRAVITY = 2  # pixel/second^2
MAX_JUMP_CHARGE = 2  # The number of time the object can jump
INITIAL_DX = -1
MOUTH_RATIO_LOWER_THRESHOLD = 0.4
# The jump speed is the facial feature score times this factor. So fo example if the facial feature is ratio
# between mouth height and mouth width, then the jump speed is ratio * JUMP_SPEED_FACTOR pixels per second.
JUMP_SPEED_FACTOR = 10

# The gravity factor works similarly to JUMP_SPEED_FACTOR. The gravity is decreased by this factor when feature score
# exceeds the lower threshold, so that when the mouth opens larger, Deafy falls slower.
GRAVITY_FACTOR = 1

assert SCREEN_HEIGHT % BACKGROUND_OBJECT_HEIGHT == 0 and SCREEN_WIDTH % BACKGROUND_OBJECT_WIDTH == 0

main_dir = os.path.split(os.path.abspath(__file__))[0]

def load_image(file, width_height = None, flip_x = False, flip_y = False):
    "loads an image, prepares it for play"
    file = os.path.join(main_dir, 'data', file)
    try:
        surface = pygame.image.load(file)
        if width_height is not None:
            surface = pygame.transform.scale(surface, width_height)
        if flip_x or flip_y:
            surface = pygame.transform.flip(surface, flip_x, flip_y)
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

class MainScreen(object):

    size = ( SCREEN_WIDTH, SCREEN_HEIGHT )
    def __init__(self, **argd):
        self.__dict__.update(**argd)
        super(MainScreen, self).__init__(**argd)

        # create a display surface. standard pygame stuff
        self.display = pygame.display.set_mode( self.size, 0 )
        self.background = pygame.Surface(SCREEN_RECT.size)

        # Initialize camera
        if ARGS.camera:
            self.init_cams(0)

        deafy_sheet = SpriteSheet("data/Undertale_Annoying_Dog.png")
        cat_sheet = SpriteSheet("data/cat.png")

        Deafy.images = [deafy_sheet.image_at((2, 101, 22-2, 119-101), colorkey=-1, width_height=(20*2,18*2), flip_x=True),
                        deafy_sheet.image_at((2, 204, 26-2, 216-204), colorkey=-1, width_height=(24*2,12*2), flip_x=True),
                        deafy_sheet.image_at((2, 182, 23-2, 200-182), colorkey=-1, width_height=(21*2,18*2), flip_x=True),
                        deafy_sheet.image_at((25, 182, 44-25, 200-182), colorkey=-1, width_height=(19*2,18*2), flip_x=True),]
        Sky.images =  [load_image('sky.png', (32,32))]
        Ground.images =  [load_image('grass.png', (32,32))]
        CatObstacle.images = [cat_sheet.image_at((0, 0, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1, 158, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1+54, 158, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1+54*2, 158, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1+54*3, 158, 54, 42), colorkey=-1),]
        # Initialize Game Groups
        self.all = pygame.sprite.RenderUpdates()
        self.background_group = pygame.sprite.RenderUpdates()
        # Sprites in this group are rendered after background so that they appear on the top.
        self.front_group = pygame.sprite.RenderUpdates()

        # assign default groups to each sprite class
        Deafy.containers = self.all, self.front_group
        Ground.containers = self.all, self.background_group
        Sky.containers = self.all, self.background_group
        CatObstacle.containers = self.all, self.front_group

        # The y goes from top to bottom starting at 0.
        self.ground_y_limits = (SCREEN_HEIGHT*3/4,SCREEN_HEIGHT)
        self.sky_y_limits = (0,SCREEN_HEIGHT*3/4)
        # TODO: Maybe the height and width are the other way around
        self.ground_sprites = [Ground(pos=(w*BACKGROUND_OBJECT_WIDTH, h*BACKGROUND_OBJECT_HEIGHT))
                               for w in range(SCREEN_WIDTH / BACKGROUND_OBJECT_WIDTH + 1)
                               for h in range(self.ground_y_limits[0] / BACKGROUND_OBJECT_HEIGHT + 1,
                                              self.ground_y_limits[1] / BACKGROUND_OBJECT_HEIGHT + 1)]
        self.sky_sprites = [Sky(pos=(w*BACKGROUND_OBJECT_WIDTH, h*BACKGROUND_OBJECT_HEIGHT))
                               for w in range(SCREEN_WIDTH / BACKGROUND_OBJECT_WIDTH + 1)
                               for h in range(self.sky_y_limits[0] / BACKGROUND_OBJECT_HEIGHT + 1,
                                              self.sky_y_limits[1] / BACKGROUND_OBJECT_HEIGHT + 1)]
        self.deafy = Deafy(pos=DEAFY_SCREEN_POS)
        self.cat_obstacles = []

        # Now initialize the FacialLandmarkDetector
        self.fld = FacialLandmarkDetector(SCREEN_WIDTH,SCREEN_HEIGHT,FACIAL_LANDMARK_PREDICTOR_WIDTH)
        self.dx = INITIAL_DX

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


    def main(self,):
        going = True
        self.clock = pygame.time.Clock()
        while going:
            events = pygame.event.get()
            for e in events:
                if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                    going = False
                if e.type == KEYDOWN:
                    if e.key == K_RIGHT:
                        self.dx -= 1
                        if self.dx < 0:
                            self.deafy.start_running()
                        # Decrease the speed for all background sprites, so it looks like deafy is moving to the right.
                        for s in self.ground_sprites  + self.cat_obstacles:
                            s.plus_dx(-2)
                        for s in self.sky_sprites:
                            s.plus_dx(-1)
                    if e.key == K_LEFT:
                        self.dx += 1
                        if self.dx >= 0:
                            self.deafy.stop_running()
                        # Increase the speed for all background sprites, so it looks like deafy is moving to the left.
                        for s in self.ground_sprites + self.cat_obstacles:
                            s.plus_dx(2)
                        for s in self.sky_sprites:
                            s.plus_dx(1)
                    if e.key == K_UP:
                        # Jump!
                        if self.deafy.is_lying:
                            self.deafy.stand_up()
                        else:
                            self.deafy.jump()
                    if e.key == K_DOWN:
                        if self.dx == 0:
                            # Lie down.
                            self.deafy.lie_down()
                    if e.key == K_c:
                        # Generate a cat
                        self.cat_obstacles.append(CatObstacle(dx=self.dx,pos=(SCREEN_WIDTH,DEAFY_SCREEN_POS[1])))

            if ARGS.camera:
                self.get_camera_shot()
                # Now use the facial landmark defector
                # This step decreases the frame rate from 30 fps to 6fps. So we need to do something about it.
                # The speed is directly related to FACIAL_LANDMARK_PREDICTOR_WIDTH.
                face_coordinates_list, facial_features_list = self.fld.get_features(self.camera_shot_raw)
                if len(face_coordinates_list) > 0:
                    print("Detected %d face%s." %(len(face_coordinates_list),
                          "s" if len(face_coordinates_list) > 1 else ""))
                    # Assume the first face is the target for now.
                    mouth_open_degree = get_mouth_open_degree(facial_features_list[0])
                    if mouth_open_degree >= MOUTH_RATIO_LOWER_THRESHOLD:
                        self.deafy.set_gravity(GRAVITY - mouth_open_degree * GRAVITY_FACTOR)
                        self.deafy.jump(mouth_open_degree * JUMP_SPEED_FACTOR)
                    print("Mouth open degree: %f" %(mouth_open_degree))


            # clear/erase the last drawn sprites
            self.all.clear(self.display, self.background)
            self.all.update()
            # # draw the scene
            dirty = self.background_group.draw(self.display)
            pygame.display.update(dirty)
            dirty = self.front_group.draw(self.display)
            pygame.display.update(dirty)
            if ARGS.camera:
                self.blit_camera_shot(self.camera_default_display_location)


            # dirty = self.all.draw(self.display)
            # pygame.display.update(dirty)
            pygame.display.flip()
            self.clock.tick(MAX_FPS)
            print (self.clock.get_fps())

class Deafy(pygame.sprite.Sprite):

    images = []
    _DEAFY_STAND_IMAGE_INDEX = 0
    _DEAFY_LIE_DOWN_IMAGE_INDEX = 1
    _DEAFY_RUN_IMAGE_START_INDEX = 2
    _DEAFY_RUN_IMAGE_END_INDEX = 3
    def __init__(self, pos=SCREEN_RECT.bottomright):
        # Notice that bottomright instead of bottomleft is used for deafy, because deafy is facing right.
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.current_image_index = self._DEAFY_RUN_IMAGE_START_INDEX
        self.image = self.images[self.current_image_index]
        self.rect = self.image.get_rect(bottomright=pos)
        self.y_speed = 0
        self.ground_level = pos[1]
        self.is_jumping = False
        self.jump_charge = MAX_JUMP_CHARGE  # The number of time the object can jump
        self.is_lying = False
        self.is_running = (INITIAL_DX < 0)
        self.gravity = GRAVITY

    def move(self, pos):
        self.rect= self.image.get_rect(bottomright=pos)
        self.rect = self.rect.clamp(SCREEN_RECT)

    def jump(self, speed=None):
        if self.jump_charge > 0:
            self.is_jumping = True
            # if the object was falling too fast, make the second jump weaker but still allow it to jump.

            if speed is None:
                d_jump_speed = random.randrange(10,50)# Give a random change in jump speed.
            else:
                d_jump_speed = speed

            self.y_speed = max(10, self.y_speed + d_jump_speed)
            self.jump_charge -= 1
        else:
            print("Not enough jump charge.")  # For debugging.

    def update(self):
        if self.is_jumping:
            self.y_speed -= self.gravity
            self.rect.move_ip(0,-self.y_speed)
            # Because y goes from top to bottom.
            if self.rect.bottom > self.ground_level:
                self.rect.bottom = self.ground_level
                self.is_jumping = False
                self.jump_charge = MAX_JUMP_CHARGE
        if self.is_running:
            self.run_next_frame()

    def change_image(self, new_image_index):
        if self.current_image_index != new_image_index:
            self.current_image_index = new_image_index
            self.image = self.images[self.current_image_index]
            self.rect = self.image.get_rect(bottomright=self.rect.bottomright)

    def lie_down(self):
        if not self.is_lying and not self.is_jumping:
            self.is_lying = True
            self.change_image(self._DEAFY_LIE_DOWN_IMAGE_INDEX)

    def stand_up(self):
        if self.is_lying:
            self.is_lying = False
            self.change_image(self._DEAFY_STAND_IMAGE_INDEX)

    def start_running(self):
        # if self.is_lying:
        #     print("Warning! Deafy attempted to start_running while it's lying down.")
        # else:
        #     self.is_running = True
        self.is_running = True
        self.change_image(self._DEAFY_RUN_IMAGE_START_INDEX)

    def stop_running(self):
        self.is_running = False
        self.change_image(self._DEAFY_STAND_IMAGE_INDEX)

    def run_next_frame(self):
        new_image_index = self.current_image_index + 1
        if new_image_index > self._DEAFY_RUN_IMAGE_END_INDEX:
            new_image_index = self._DEAFY_RUN_IMAGE_START_INDEX
        self.change_image(new_image_index)

    def set_gravity(self, new_gravity):
        """
        Modifies self.gravity to be the new gravity. Gravity cannot be lower than 0.
        :param new_gravity:
        :return: Nothing
        """
        self.gravity = max(0,new_gravity)


class BackgroundObjects(pygame.sprite.Sprite):
    images = []
    def __init__(self, dx=INITIAL_DX, pos=SCREEN_RECT.bottomleft, destroy_when_oos=False):
        """

        :param pos: The initial position of the object.
        :param destroy_when_oos: If true, the object self destroys when it is out of the screen. If False, the object
        wraps around the screen when it is out of the screen.
        """
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.current_image_index = 0
        self.image = self.images[self.current_image_index]
        self.rect = self.image.get_rect(bottomleft=pos)
        self.dx = dx
        self.destroy_when_oos=destroy_when_oos


    def update(self):
        # Makes the enemy move in the x direction.
        self.rect.move_ip(self.dx, 0)

        # If the enemy is outside of the platform, make it appear on the other side of the screen
        if self.rect.left > SCREEN_RECT.right or self.rect.right < SCREEN_RECT.left:
            if self.destroy_when_oos:
                self.kill()
                return
            else:
                if self.rect.left > SCREEN_RECT.right:
                    # Move the sprite towards the left n pixels where n = width of platform + width of object.
                    self.rect.move_ip(SCREEN_RECT.left-SCREEN_RECT.right - BACKGROUND_OBJECT_WIDTH,0)
                elif self.rect.right < SCREEN_RECT.left:
                    # Move the sprite towards the right n pixels where n = width of platform + width of object.
                    self.rect.move_ip(SCREEN_RECT.right-SCREEN_RECT.left + BACKGROUND_OBJECT_WIDTH,0)
                else:
                    raise AssertionError("This line should not be reached. The object should only move left and right.")


    def set_dx(self, dx):
        self.dx = dx

    def plus_dx(self, ddx):
        self.dx += ddx

    def change_image(self, new_image_index):
        if self.current_image_index != new_image_index:
            self.current_image_index = new_image_index
            self.image = self.images[self.current_image_index]
            self.rect = self.image.get_rect(bottomleft=self.rect.bottomleft)


class Ground(BackgroundObjects):
    pass


class Sky(BackgroundObjects):
    pass


class CatObstacle(BackgroundObjects):
    _CAT_SIT_IMAGE_INDEX = 0
    _CAT_RUN_IMAGE_START_INDEX = 1
    _CAT_RUN_IMAGE_END_INDEX = 4

    def __init__(self, dx=INITIAL_DX, pos=SCREEN_RECT.bottomleft, destroy_when_oos=True):
        super(CatObstacle, self).__init__(dx, pos,destroy_when_oos)
        if len(self.images) < (self._CAT_RUN_IMAGE_END_INDEX + 1):
            raise AssertionError("Wrong number of images loaded for class CatObstacle. "
                                 "It should be more than %d but it is now %d"
                                 %((self._CAT_RUN_IMAGE_END_INDEX + 1), len(self.images)))

    def update(self):
        super(CatObstacle, self).update()
        if self.dx < 0:
            # is moving
            self.run_next_frame()
        else:
            self.change_to_sit_frame()

    def run_next_frame(self):
        new_image_index = self.current_image_index + 1
        if new_image_index > self._CAT_RUN_IMAGE_END_INDEX:
            new_image_index = self._CAT_RUN_IMAGE_START_INDEX
        self.change_image(new_image_index)

    def change_to_sit_frame(self):
        self.change_image(self._CAT_SIT_IMAGE_INDEX)

def main():
    pygame.init()
    pygame.camera.init()

    MainScreen().main()
    pygame.quit()

if __name__ == '__main__':
    main()
