#!/usr/bin/env python

# This file builds a simple 2d arcade game.

import argparse
import os
import random
import time

import pygame
import pygame.camera
from pygame.locals import *

from facial_landmark_util import FacialLandmarkDetector, get_mouth_open_score
from sprite_sheet import SpriteSheet


# Command line argument parser.
parser = argparse.ArgumentParser()

parser.add_argument("--no_camera", dest="camera", action="store_false",
                    help="Turn off camera and use keyboard control.")
parser.set_defaults(camera=True)

ARGS = parser.parse_args()

# game constants
SCREEN_HEIGHT=480
SCREEN_WIDTH=640
SCREEN_RECT= Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
FACIAL_LANDMARK_PREDICTOR_WIDTH = 320
CAMERA_INPUT_HEIGHT = 480
CAMERA_INPUT_WIDTH = 640
CAMERA_DISPLAY_HEIGHT = SCREEN_HEIGHT / 4
CAMERA_DISPLAY_WIDTH = SCREEN_WIDTH / 4
CAMERA_DISPLAY_SIZE = (CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT)
BACKGROUND_OBJECT_HEIGHT = 32
BACKGROUND_OBJECT_WIDTH = 32
MAX_JUMP_CHARGE = 2  # The number of time the object can jump
DIALOG_FRAME_COUNT = 4 # The number of total dialog frames


# UI object specifics
# The y goes from top to bottom starting at 0.
GROUND_LEVEL = SCREEN_HEIGHT*11/15
GROUND_Y_LIMITS = (GROUND_LEVEL,SCREEN_HEIGHT)
SKY_Y_LIMITS = (0,GROUND_LEVEL)
DEAFY_SCREEN_POS = (SCREEN_WIDTH/8, GROUND_LEVEL)

MAX_FPS = 30
INITIAL_GRAVITY = 2  # pixel/second^2
MAX_JUMP_CHARGE = 1  # The number of time the object can jump
INITIAL_DX = 0
STD_DX = -1
MOUTH_SCORE_JUMP_LOWER_THRESHOLD = 0.6  # The mouth score has to be at least this big for Deafy to start jumping.
# The jump speed is the facial feature score times this factor. So fo example if the facial feature is ratio
# between mouth height and mouth width, then the jump speed is ratio * BLINK_JUMP_SPEED_FACTOR pixels per second.
JUMP_SPEED_FACTOR = 30
# The gravity factor works similarly to BLINK_JUMP_SPEED_FACTOR. The gravity is decreased by this factor when feature score
# exceeds the lower threshold, so that when the mouth opens larger, Deafy falls slower.
GRAVITY_FACTOR = 1
MIN_GRAVITY = 2.0 # 0.5
# This controls the speed at which Deafy moves. (Temporary solution for the demo)
MOUTH_SCORE_SPEED_THRESHOLDS = [(0.3, -2), (0.4, -4), (0.6, -6), (0.8, -8)]


MOUTH_LEFT_CORNER_THRESHOLD = 0.8

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
    # Note: if the sound does not play, convert it to .ogg format using this
    # website https://www.onlinevideoconverter.com/convert-wav-to-mp3
    # Or use sudo apt-get install vorbis-tools And to encode: oggenc -q 3 -o file.ogg file.wav
    if not pygame.mixer: return dummysound()
    file = os.path.join(main_dir, 'data', file)
    try:
        sound = pygame.mixer.Sound(file)
        return sound
    except pygame.error:
        print ('Warning, unable to load, %s' % file)
    return dummysound()


class Item(object):

    def __init__(self, xstart, height, width, type):
        self.xstart = xstart
        self.height = height
        self.width = width
        self.type = type


class Stage(object):

    _FINISH_LINE_DIST = 100

    def __init__(self, num, **argd):
        self.stage_num = num
        path = 'data/stages/' + str(num) + '.stage'
        with open(path, 'r') as f:
            stage_layout = f.read()
        stage_layout = stage_layout.split('\n')
        stage_layout = filter(lambda n: n != '', stage_layout)
        if len(stage_layout) == 0:
            raise AssertionError("Cannot read info about at path "+path)
        stage_layout = map(lambda n: map(int, n.split(' ')), stage_layout)
        stage_layout.sort(key=lambda n:n[0])
        self.stage_layout = []
        for s in stage_layout:
            self.stage_layout.append(Item(s[0], s[1], s[2], s[3]))
        self.next = 0
        self.length = len(self.stage_layout)
        last = self.stage_layout[-1]
        self.finish_line = last.xstart + last.width + self._FINISH_LINE_DIST

    def view_next_item(self):
        if self.next == self.length:
            return None
        return self.stage_layout[self.next]

    def pop_next_item(self):
        if self.next == self.length:
            return None
        obstacle = self.stage_layout[self.next]
        self.next += 1
        return obstacle

    def checkWin(self, x):
        return x > self.finish_line


class MainScreen(object):

    size = ( SCREEN_WIDTH, SCREEN_HEIGHT )
    def __init__(self, **argd):
        self.__dict__.update(**argd)
        super(MainScreen, self).__init__(**argd)

        # create a display image. standard pygame stuff
        self.display = pygame.display.set_mode( self.size, 0 )
        self.background = pygame.Surface(SCREEN_RECT.size)

        # Initialize camera
        if ARGS.camera:
            self.init_cams(0)

        # Load graphics
        deafy_sheet = SpriteSheet("data/Undertale_Annoying_Dog.png")
        cat_sheet = SpriteSheet("data/cat.png")

        Deafy.images = [deafy_sheet.image_at((2, 101, 22-2, 119-101), colorkey=-1, width_height=(20*2,18*2), flip_x=True),
                        deafy_sheet.image_at((2, 204, 26-2, 216-204), colorkey=-1, width_height=(24*2,12*2), flip_x=True),
                        deafy_sheet.image_at((2, 182, 23-2, 200-182), colorkey=-1, width_height=(21*2,18*2), flip_x=True),
                        deafy_sheet.image_at((25, 182, 44-25, 200-182), colorkey=-1, width_height=(19*2,18*2), flip_x=True),
                        deafy_sheet.image_at((2, 101, 22-2, 119-101), colorkey=-1, width_height=(20 * 2, 18 * 2),
                                             flip_x=True, flip_y=True),]
        Sky.images =  [load_image('sky.png', (32,32))]
        Ground.images =  [load_image('grass.png', (32,32))]
        GroundObstacle.images = [load_image('grass.png', (32,32)), load_image('sky.png', (32,32))]
        CatObstacle.images = [cat_sheet.image_at((0, 0, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1, 158, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1+54, 158, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1+54*2, 158, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1+54*3, 158, 54, 42), colorkey=-1),]
        # Load sounds
        Deafy.sounds = [load_sound("normal.ogg"), load_sound("jump.ogg"), load_sound("victory.ogg")]

        # Initialize Game Groups
        self.all = pygame.sprite.RenderUpdates()
        self.background_group = pygame.sprite.RenderUpdates()
        self.obstacle_group = pygame.sprite.RenderUpdates()
        # Sprites in this group are rendered after background so that they appear on the top.
        self.front_group = pygame.sprite.RenderUpdates()

        # assign default groups to each sprite class
        Deafy.containers = self.all, self.front_group
        Ground.containers = self.all, self.background_group
        Sky.containers = self.all, self.background_group
        GroundObstacle.containers = self.all, self.obstacle_group
        CatObstacle.containers = self.all, self.front_group
        Dialog.containers = self.all

        # initialize stage
        self.stage = Stage(num=1)
        self.current_items = []

        # TODO: Maybe the height and width are the other way around
        self.ground_sprites = [Ground(pos=(w*BACKGROUND_OBJECT_WIDTH, h*BACKGROUND_OBJECT_HEIGHT))
                               for w in range(SCREEN_WIDTH / BACKGROUND_OBJECT_WIDTH + 1)
                               for h in range(GROUND_Y_LIMITS[0] / BACKGROUND_OBJECT_HEIGHT + 1,
                                              GROUND_Y_LIMITS[1] / BACKGROUND_OBJECT_HEIGHT + 1)]
        self.sky_sprites = [Sky(pos=(w*BACKGROUND_OBJECT_WIDTH, h*BACKGROUND_OBJECT_HEIGHT))
                               for w in range(SCREEN_WIDTH / BACKGROUND_OBJECT_WIDTH + 1)
                               for h in range(SKY_Y_LIMITS[0] / BACKGROUND_OBJECT_HEIGHT + 1,
                                              SKY_Y_LIMITS[1] / BACKGROUND_OBJECT_HEIGHT + 1)]
        self.deafy = Deafy(pos=DEAFY_SCREEN_POS)
        self.ground_obstacle_sprites = []
        self.cat_obstacles = []

        # Now initialize the FacialLandmarkDetector
        self.fld = FacialLandmarkDetector(SCREEN_WIDTH,SCREEN_HEIGHT,FACIAL_LANDMARK_PREDICTOR_WIDTH)
        self.dx = INITIAL_DX
        self.visible_xrange = [0, SCREEN_WIDTH]

        # to track of which dialog frame shold be rendered
        self.dialog_frame = 0
        # To trach whether dialog is displayed now. If so, disable user control.
        self.is_dialog_active = True

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

        # create a image to capture to.  for performance purposes, you want the
        # bit depth to be the same as that of the display image.
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
        #     # blit it to the display image.  simple!
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

    def set_dx(self, new_dx):
        """
        This is a function to control overall speed.
        :return:
        """
        self.dx = new_dx
        print self.dx
        if self.dx < 0:
            self.deafy.start_running()
        else:
            self.deafy.stop_running()
        for s in self.sky_sprites + self.ground_sprites + self.cat_obstacles + self.ground_obstacle_sprites:
            s.set_dx(self.dx)

    def change_dx(self, change):
        """
        This is a function to control overall speed.
        :return:
        """
        self.dx += change
        print self.dx
        if self.dx < 0:
            self.deafy.start_running()
        else:
            self.deafy.stop_running()
        for s in self.sky_sprites + self.ground_sprites + self.cat_obstacles + self.ground_obstacle_sprites:
            s.set_dx(self.dx)

    def handle_game_over(self):
        print 'Sorry, Game Over! :( '
        return

    def main(self,):

        going = True
        self.clock = pygame.time.Clock()
        while going:
            events = pygame.event.get()
            for e in events:
                if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                    going = False
                if e.type == KEYDOWN:
                    # handle different keys
                    if e.key == K_RIGHT:
                        self.change_dx(-1)
                    if e.key == K_LEFT:
                        if self.dx < 0:
                            self.change_dx(1)
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
                    if e.key == K_SPACE:
                        if self.dialog_frame < DIALOG_FRAME_COUNT:
                            self.dialog_frame += 1
                            if self.dialog_frame >= DIALOG_FRAME_COUNT:
                                self.is_dialog_active = False


            if ARGS.camera and not self.is_dialog_active:
                self.get_camera_shot()
                # Now use the facial landmark defector
                # This step decreases the frame rate from 30 fps to 6fps. So we need to do something about it.
                # The speed is directly related to FACIAL_LANDMARK_PREDICTOR_WIDTH.
                face_coordinates_list, facial_features_list = self.fld.get_features(self.camera_shot_raw)
                if len(face_coordinates_list) > 0:
                    print("Detected %d face%s." %(len(face_coordinates_list),
                          "s" if len(face_coordinates_list) > 1 else ""))
                    # Assume the first face is the target for now.
                    mouth_open_score = get_mouth_open_score(facial_features_list[0])
                    if mouth_open_score >= MOUTH_SCORE_JUMP_LOWER_THRESHOLD:
                        self.deafy.set_gravity(INITIAL_GRAVITY - mouth_open_score * GRAVITY_FACTOR)
                        self.deafy.jump(mouth_open_score * JUMP_SPEED_FACTOR)
                    # Now set the speed for deafy based on how open the mouth is.
                    for threshold_i, (threshold, speed) in enumerate(MOUTH_SCORE_SPEED_THRESHOLDS):
                        if mouth_open_score >= threshold:
                            self.set_dx(speed)
                        else:
                            if threshold_i == 0:
                                # Mouth score is smaller than the first threshold. Set dx to 0.
                                self.set_dx(0)
                            break
                    print("Mouth open degree: %f" %(mouth_open_score))
                    # # Use the left mouth corner to decrease the speed
                    # mouth_left_corner_score = get_mouth_left_corner_score(facial_features_list[0])
                    # if mouth_left_corner_score >= MOUTH_LEFT_CORNER_THRESHOLD:
                    #     self.change_dx(-1)
                    # print("Mouth left corner score: %f" %(mouth_left_corner_score))
                else:
                    # TODO: maybe add a smoothing factor. Otherwise Deafy stops whenever the camera cannot detect the
                    # face, making the game harder to control.
                    self.set_dx(0)
                    self.deafy.set_gravity(INITIAL_GRAVITY)

            # based on dx, update the current screen state
            self.visible_xrange = map(lambda n:n-self.dx, self.visible_xrange)
            self.current_items = filter(lambda item:item.xstart + item.width >= self.visible_xrange[0], self.current_items)
            # if self.dx != 0:
            #     print self.visible_xrange
            while self.stage.view_next_item() and self.stage.view_next_item().xstart < self.visible_xrange[1]:
                item = self.stage.pop_next_item()
                self.current_items.append(item)
                print self.visible_xrange, self.current_items
                # create ui object for this item
                xstart = item.xstart - self.visible_xrange[0]
                if item.type == 0:
                    ystart = GROUND_LEVEL - item.height
                if item.type == 1:
                    ystart = GROUND_LEVEL
                ystart += BACKGROUND_OBJECT_HEIGHT
                # how to show half sprites???
                self.ground_obstacle_sprites.extend([
                    GroundObstacle(item_type=item.type, dx=self.dx, pos=(w, h))
                    for w in range(xstart, xstart+item.width, BACKGROUND_OBJECT_WIDTH)
                    for h in range(ystart, ystart+item.height, BACKGROUND_OBJECT_HEIGHT)])


            # update the current game state
            # update the ground level on the current screen
            ground_level = [GROUND_LEVEL for i in xrange(SCREEN_WIDTH)]
            for item in self.current_items:
                item_left = item.xstart - self.visible_xrange[0]
                for x in xrange(max(0, item_left), min(SCREEN_WIDTH, item_left+item.width)):
                    if item.type == 0:
                        ground_level[x] -= item.height
                    elif item.type == 1:
                        ground_level[x] += item.height
            # if the ground level in front of deafy is higher than itself, stop running.
            dleft, dright = self.deafy.rect.left, self.deafy.rect.right
            dbottom, dy = self.deafy.rect.bottom, self.deafy.y_speed
            for x in xrange(dright, dright-self.dx):
                if self.deafy.rect.bottom > ground_level[x]:
                    self.set_dx(0)  # TODO: also disable set_dx() anywhere else.
                    self.deafy.failed = True  # Disable user control on deafy afterwards.
            # check the dog's motion with respect to the ground
            valid_ground_level = min(ground_level[dleft:dright])    # use the highest ground level
            if dbottom <= valid_ground_level <= dbottom-dy:
                if self.deafy.y_speed <= 0:
                    self.deafy.land_on_ground(ground=valid_ground_level)
            elif dbottom-dy < valid_ground_level:
                self.deafy.fall()

            # check the status of the game (win / lose)
            # winning: if the dog succeeded going 100 pixel further than the last stage item
            # TODO: add a flag or a person or something to indicate end point, after the demo.
            if self.stage.checkWin(self.deafy.rect.left+self.visible_xrange[0]):
                print 'You win!'
                self.deafy.play_sound(self.deafy._DEAFY_VICTORY_SOUND_INDEX)
                time.sleep(5)  # TODO: Sleep this amount of seconds to make sure sound finishes playing. Improve this part after the demo to add victory screen and fail screen.
                going = False
            # lose if the dog falls completely outside the screen
            if self.deafy.rect.top > SCREEN_HEIGHT:
                self.handle_game_over()
                going = False


            # clear/erase the last drawn sprites
            self.all.clear(self.display, self.background)
            self.all.update()
            # # draw the scene
            dirty = self.background_group.draw(self.display)
            pygame.display.update(dirty)
            dirty = self.obstacle_group.draw(self.display)
            pygame.display.update(dirty)
            dirty = self.front_group.draw(self.display)
            pygame.display.update(dirty)

            # display dialog
            if self.is_dialog_active:
                # TODO: minor detail but it might be better to keep one single dialog object instead of creating a
                # new object every time. So like self.dialog.update_frame(self.dialog_frame) or something.
                self.dialog = Dialog(self.dialog_frame)
                # TODO: maybe use the self.rect and self.update instead. That is the standard way to display a sprite.
                # Now the display blit is handled manually. Add it to a group and use methods like above to make sure
                # it is drawn after everything else. The blinking is likely caused by this bug.
                self.display.blit(self.dialog.image, (SCREEN_WIDTH-320, SCREEN_HEIGHT-120))

            # enable camera only after all dialog frames are shown
            if ARGS.camera and not self.is_dialog_active:
                self.blit_camera_shot(self.camera_default_display_location)

            # dirty = self.all.draw(self.display)
            # pygame.display.update(dirty)
            pygame.display.flip()
            self.clock.tick(MAX_FPS)
            # print (self.clock.get_fps())

class Deafy(pygame.sprite.Sprite):

    images = []
    sounds = []
    _DEAFY_STAND_IMAGE_INDEX = 0
    _DEAFY_LIE_DOWN_IMAGE_INDEX = 1
    _DEAFY_RUN_IMAGE_START_INDEX = 2
    _DEAFY_RUN_IMAGE_END_INDEX = 3
    _DEAFY_FAIL_INDEX = 4
    _DEAFY_JUMP_SOUND_INDEX = 1
    _DEAFY_VICTORY_SOUND_INDEX = 2
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
        self.gravity = INITIAL_GRAVITY
        self.failed = False  # If true, disable user control.

    def move(self, pos):
        self.rect= self.image.get_rect(bottomright=pos)
        self.rect = self.rect.clamp(SCREEN_RECT)

    def jump(self, speed=None):
        if self.failed:
            return
        if self.jump_charge > 0:
            self.is_jumping = True
            # if the object was falling too fast, make the second jump weaker but still allow it to jump.

            if speed is None:
                d_jump_speed = random.randrange(10,50)# Give a random change in jump speed.
            else:
                d_jump_speed = speed

            self.y_speed = max(10, self.y_speed + d_jump_speed)
            self.jump_charge -= 1
            # Play sound effect
            self.play_sound(self._DEAFY_JUMP_SOUND_INDEX)
        else:
            print("Not enough jump charge.")  # For debugging.

    def fall(self):
        if not self.is_jumping:
            self.is_jumping = True
            self.jump_charge = 0

    def update(self):
        if self.is_jumping:
            self.y_speed -= self.gravity
            self.rect.move_ip(0,-self.y_speed)
        if self.is_running:
            self.run_next_frame()

    def land_on_ground(self, ground):
        self.rect.bottom = ground
        self.is_jumping = False
        self.y_speed = 0
        self.jump_charge = MAX_JUMP_CHARGE

    def change_image(self, new_image_index):
        if self.current_image_index != new_image_index:
            self.current_image_index = new_image_index
            self.image = self.images[self.current_image_index]
            self.rect = self.image.get_rect(bottomright=self.rect.bottomright)

    def play_sound(self, sound_index):
        if sound_index >= len(self.sounds):
            raise IndexError("Sound index %d exceeding number of sounds stored (%d)." %(sound_index, len(self.sounds)))
        self.sounds[sound_index].play()

    def lie_down(self):
        if self.failed:
            return
        if not self.is_lying and not self.is_jumping:
            self.is_lying = True
            self.change_image(self._DEAFY_LIE_DOWN_IMAGE_INDEX)

    def stand_up(self):
        if self.failed:
            return
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
        # Only stop animation when it's not jumping.
        if not self.is_jumping:
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
        self.gravity = max(MIN_GRAVITY,new_gravity)



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
                self.handle_oos()
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

    def handle_oos(self):
        if self.destroy_when_oos:
            self.kill()


class Ground(BackgroundObjects):
    pass


class Sky(BackgroundObjects):
    pass



class GroundObstacle(BackgroundObjects):

    _GROUND_IMAGE_INDEX = 0
    _GAP_IMAGE_INDEX = 1

    def __init__(self, item_type, dx=INITIAL_DX, pos=SCREEN_RECT.bottomleft, destroy_when_oos=True):
        super(GroundObstacle, self).__init__(dx, pos, destroy_when_oos)
        if item_type == self._GROUND_IMAGE_INDEX or item_type == self._GAP_IMAGE_INDEX:
            self.change_image(item_type)

    def update(self):
        before = self.rect.left, self.rect.right
        super(GroundObstacle, self).update()
        after = self.rect.left, self.rect.right
        # if before != after:
        #     print before, after

    def handle_oos(self):
        # only kill the obstacle when it reaches the far left
        if self.rect.right < SCREEN_RECT.left:
            self.kill()



class CatObstacle(BackgroundObjects):
    _CAT_SIT_IMAGE_INDEX = 0
    _CAT_RUN_IMAGE_START_INDEX = 1
    _CAT_RUN_IMAGE_END_INDEX = 4

    def __init__(self, dx=STD_DX, pos=SCREEN_RECT.bottomleft, destroy_when_oos=True):
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

class Dialog(pygame.sprite.Sprite):
    width = 300
    height = 100
    white = (255,255,255)
    black = (0,0,0)

    text_group = []
    texts = ['Deafy is lost in the wild and he needs', 'your help. He can not hear']
    texts2 = ['but has a sharp vision. Use your facial ', 'expression, specifically']
    texts3 = ['the extent in which you open your ', 'mouth to control Deafy\'s running']
    texts4 = ['and running and help guide him home.', ' ']
    text_group.append(texts)
    text_group.append(texts2)
    text_group.append(texts3)
    text_group.append(texts4)

    def __init__(self, diglog_index):
        pygame.font.init()
        myfont = pygame.font.SysFont('Comic Sans MS', 22)

        self.image = pygame.Surface([Dialog.width, Dialog.height])
        self.image.fill(Dialog.black)
        self.rect = self.image.get_rect()

        border_width = 5
        # horizontal border
        h_border = pygame.Surface([Dialog.width, border_width])
        h_border.fill(Dialog.white)
        # vertical border
        v_border = pygame.Surface([border_width, Dialog.height])
        v_border.fill(Dialog.white)

        # render current frame text
        current_text = Dialog.text_group[diglog_index]
        distance = 0
        for i in range(len(current_text)):
            textsurface = myfont.render(current_text[i], False, Dialog.white)
            self.image.blit(textsurface, (15, 15 + distance))
            distance += (i+1) * 15
        space_indicator = myfont.render('>>> Space', False, Dialog.white)
        self.image.blit(space_indicator, (200, 15 + distance))

        # add border to each edge of the image
        self.image.blit(v_border, (0,0))
        self.image.blit(v_border, (Dialog.width-border_width,0))
        self.image.blit(h_border, (0,0))
        self.image.blit(h_border, (0, Dialog.height-border_width))


def main():
    pygame.init()
    pygame.camera.init()

    MainScreen().main()
    pygame.quit()

if __name__ == '__main__':
    main()
