#!/usr/bin/env python

# This file builds a simple 2d arcade game.

import argparse
import os
import random
import time
import numpy as np
import cv2
from multiprocessing import Process, Queue, Value

import pygame
import pygame.camera
import pygame.key
from pygame.locals import *

from facial_landmark_util import FacialLandmarkDetector, HeadPoseEstimator, get_mouth_open_score, \
    get_any_eye_blink_rounds, get_direction_from_line, get_take_photo_score
from sprite_sheet import SpriteSheet

from constants import *
from dialog import *
from bullet import *
from obstacle import *
from deafy_cat import *
from hp_bar import *
from face_illustration import *
from text import Text


# Command line argument parser.
parser = argparse.ArgumentParser()

parser.add_argument("--no_camera", dest="camera", action="store_false",
                    help="Turn off camera and use keyboard control.")
parser.set_defaults(camera=True)

parser.add_argument("--deafy_camera_index", dest="deafy_camera_index", type=int,
                    help="The camera index for Deafy player.")
parser.add_argument("--cat_camera_index", dest="cat_camera_index", type=int,
                    help="The camera index for Cat player.")
parser.add_argument("--debug_level", dest="debug_level", type=int, default=0,
                    help="0 For only printing necessary message during gameplay. "
                         "1 for printing unexpected errors that does not cause game to stop."
                         "2 and above for printing regular debugging messages.")

ARGS = parser.parse_args()

if ARGS.camera and ARGS.deafy_camera_index is None and ARGS.cat_camera_index is None:
    parser.error("To use the camera, please specify either the camera index for deafy or for cat, or both.")

DEBUG_LEVEL = ARGS.debug_level

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
        if DEBUG_LEVEL >= DEBUG_PRINT_UNEXPECTED_ERROR:
            print ('Warning, unable to load, %s' % file)
    return dummysound()

def fire_bullet(object_firing, object_orientation, bullet_speed=BULLET_SPEED, bullet_color=BLACK):
    """

    :param object_firing: The object firing the bullet. Must have the .rect attribute.
    :param object_orientation: "LEFT" or "RIGHT"
    :param bullet_speed: A positive number, it represents the speed of the bullet independent of its direction.
    :param bullet_color: A tuple representing its _HP_COLOR.
    :return:
    """
    if bullet_speed <= 0:
        raise AttributeError("Bullet speed must be positive.")
    if object_orientation == "LEFT":
        bullet_location = object_firing.rect.midleft
        # Move the bullet a little to avoid hiting the object firing the bullet.
        bullet_location = (bullet_location[0]-BULLET_SIZE, bullet_location[1])
        return Bullet(-bullet_speed, pos=bullet_location, color=bullet_color,bullet_size=BULLET_SIZE)
    else:
        bullet_location = object_firing.rect.midright
        # Move the bullet a little to avoid hiting the object firing the bullet.
        bullet_location = (bullet_location[0] + BULLET_SIZE, bullet_location[1])
        return Bullet(bullet_speed, pos=bullet_location, color=bullet_color,bullet_size=BULLET_SIZE)


class StageItem(object):

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
            self.stage_layout.append(StageItem(s[0], s[1], s[2], s[3]))
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

    size = (GAME_SCREEN_WIDTH, GAME_SCREEN_HEIGHT)

    def __init__(self, **argd):
        self.__dict__.update(**argd)
        super(MainScreen, self).__init__(**argd)

        # create a display image. standard pygame stuff
        self.display = pygame.display.set_mode( self.size, 0 )
        self.background = pygame.Surface(GAME_SCREEN_RECT.size)

        # Create text display helper class.
        self.text = Text()

        # Initialize the stop flag. When the game stops, we need to stop the subprocess as well. Set the flag to any
        # value other than 0 to stop the subprocesses.
        self.stop_flag = Value('i', 0)

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
        CatOpponent.images = [cat_sheet.image_at((0, 0, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1, 158, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1+54, 158, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1+54*2, 158, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1+54*3, 158, 54, 42), colorkey=-1), ]

        self.default_face_photo = load_image('Barack-Obama-Funny-Face-Image.jpg')
        # Load sounds
        Deafy.sounds = [load_sound("normal.ogg"), load_sound("jump.ogg"), load_sound("victory.ogg")]
        CatOpponent.sounds=[load_sound("cat_meow.wav"), load_sound("Cat-meow-nervous.wav"),
                            load_sound("Angry-cat-sound.wav")]
        self.background_sounds = [load_sound("2A03_fluidvolt-Pallid_Underbrush.wav"),
                                    load_sound("2A03_Kevvviiinnn-Superfusion.wav"),
                                    load_sound("Class_gyms-Pastorale.wav"),
                                    load_sound("Class_Kulor-SpaceDolphinsSpaceCave.wav"),
                                    load_sound("Class_Zephemeros-Asymmetrical.wav"),
                                    load_sound("FDS_Kevvviiinnn_-_The_Devourer's_Wrath.wav"),
                                    load_sound("FDS_rushjet1_-_fdx.wav"),
                                    load_sound("FDS_SriK_-_F%!&_Dis_S#!%.wav"),
                                    load_sound("miku-joker.wav"),
                                    load_sound("MMC5_moviemovies1-Rubicon.wav"),
                                    load_sound("N163_Jayster_-_TMNT_Tournament_Fighters_Scrapyard_Swing.wav"),
                                    load_sound("VRC6_Anon-3DGalax.wav"),
                                    load_sound("VRC6_ArchJ-MegaManRemixesArchj - Track 01 Mega Man Elecman.wav"),
                                    load_sound("VRC6_Ares64-smurfity smurf - Track 01 (Smurfs (GB) Title).wav"),
                                    load_sound("VRC6_Ares64-smurfity smurf - Track 02 (Smurfs (GB) Volcano).wav"),
                                    load_sound("VRC6_Ares64-smurfity smurf - Track 03 (Smurfs (GB) River Smurf).wav"),
                                    load_sound("VRC6_Raijin-Thunderforce III - Haides (Truth) Devil Crash - Main Theme - Track 01 (Thunderforce III Truth).wav"),
                                    load_sound("VRC6_Raijin-Thunderforce III - Haides (Truth) Devil Crash - Main Theme - Track 02 (Devil Crash Main Theme).wav")]
        self.background_channel = pygame.mixer.Channel(BGM_CHANNEL_ID)
        self.background_channel.set_endevent(END_BGM_EVENT)

        # Initialize Game Groups
        self.all = pygame.sprite.RenderUpdates()
        self.background_group = pygame.sprite.RenderUpdates()
        self.obstacle_group = pygame.sprite.RenderUpdates()
        # Sprites in this group are rendered after background so that they appear on the top.
        self.front_group = pygame.sprite.RenderUpdates()
        self.player_group = pygame.sprite.Group()  # Not an ui group, just for collision detection.

        # assign default groups to each sprite class
        Deafy.containers = self.all, self.front_group, self.player_group
        Ground.containers = self.all, self.background_group
        Sky.containers = self.all, self.background_group
        GroundObstacle.containers = self.all, self.obstacle_group
        CatOpponent.containers = self.all, self.front_group, self.player_group
        Bullet.containers = self.all, self.front_group
        HPBar.containers = self.all, self.background_group
        Face.containers = self.all, self.background_group
        Dialog.containers = self.all, self.background_group

        self.deafy_player_photos = []
        self.cat_player_photos = []
        # Initialize camera
        if ARGS.camera:
            # HeadPoseEstimator
            self.hpe = HeadPoseEstimator(CAMERA_INPUT_WIDTH, CAMERA_INPUT_HEIGHT)

            if ARGS.deafy_camera_index is not None:
                # Facial landmark detector.
                self.deafy_fld = FacialLandmarkDetector(CAMERA_INPUT_WIDTH, CAMERA_INPUT_HEIGHT, name="Deafy")
                self.init_cams(ARGS.deafy_camera_index, DEAFY_CAMERA_DISPLAY_LOCATION)
                self.deafy_cam_on = True
                self.deafy_player_face = Face(CAMERA_INPUT_SIZE,FACE_DEAFY_BOTTOMLEFT)
                self.deafy_queue = Queue()
                self.deafy_features_q = Queue()
            else:
                self.deafy_fld = None
                self.deafy_cam_on = False
                self.deafy_player_face = None
                self.deafy_queue = None
                self.deafy_features_q = None
            if ARGS.cat_camera_index is not None:
                # Facial landmark detector.
                self.cat_fld = FacialLandmarkDetector(CAMERA_INPUT_WIDTH, CAMERA_INPUT_HEIGHT, name="Cat")
                self.init_cams(ARGS.cat_camera_index, CAT_CAMERA_DISPLAY_LOCATION)
                self.cat_cam_on = True
                self.cat_player_face = Face(CAMERA_INPUT_SIZE, FACE_CAT_BOTTOMLEFT)
                self.cat_queue = Queue()
                self.cat_features_q = Queue()
            else:
                self.cat_fld = None
                self.cat_cam_on = False
                self.cat_player_face = None
                self.cat_queue = None
                self.cat_features_q = None

    def init_battle(self):

        # # initialize stage
        # self.stage = Stage(num=1)
        # self.current_items = []

        # TODO: Maybe the height and width are the other way around
        self.ground_sprites = [Ground(pos=(w*BACKGROUND_OBJECT_WIDTH, h*BACKGROUND_OBJECT_HEIGHT))
                               for w in range(BATTLE_SCREEN_WIDTH / BACKGROUND_OBJECT_WIDTH + 1)
                               for h in range(GROUND_Y_LIMITS[0] / BACKGROUND_OBJECT_HEIGHT + 1,
                                              GROUND_Y_LIMITS[1] / BACKGROUND_OBJECT_HEIGHT + 1)]
        self.sky_sprites = [Sky(pos=(w*BACKGROUND_OBJECT_WIDTH, h*BACKGROUND_OBJECT_HEIGHT))
                               for w in range(BATTLE_SCREEN_WIDTH / BACKGROUND_OBJECT_WIDTH + 1)
                               for h in range(SKY_Y_LIMITS[0] / BACKGROUND_OBJECT_HEIGHT + 1,
                                              SKY_Y_LIMITS[1] / BACKGROUND_OBJECT_HEIGHT + 1)]
        self.deafy = Deafy(pos=DEAFY_SCREEN_POS)
        self.cat = CatOpponent(pos=CAT_SCREEN_POS)
        self.ground_obstacle_sprites = []
        self.cat_obstacles = []
        self.bullets = []
        self.dialog = Dialog(DIALOG_FRAME_COUNT)

        self.dx = INITIAL_DX
        self.visible_xrange = [0, BATTLE_SCREEN_WIDTH]

        # to track of which dialog frame shold be rendered
        self.dialog_frame = 0
        # To trach whether dialog is displayed now. If so, disable user control
        self.is_dialog_active = True # Disabled for demo purpose. Maybe add back later.
        self.dialog_space_recharged = True

        # Reset sound and play a new random background music.
        self.background_channel.stop()
        self._queue_random_bgm()


    def reset_battle(self):
        # I know this is weird but it seems to be able to avoid some lag and make sure no
        # keyboard inputs are carried over to the next game
        time.sleep(1)
        events = pygame.event.get()
        time.sleep(1)
        events = pygame.event.get()
        time.sleep(1)
        events = pygame.event.get()
        self.deafy.kill()
        self.cat.kill()
        self.deafy = Deafy(pos=DEAFY_SCREEN_POS)
        self.cat = CatOpponent(pos=CAT_SCREEN_POS)
        for b in self.bullets:
            b.kill()
        self.bullets = []

        # to track of which dialog frame shold be rendered
        self.dialog_frame = 0
        # To trach whether dialog is displayed now. If so, disable user control.
        self.is_dialog_active = False # Disabled for demo purpose. Maybe add back later.

        # Reset sound and play a new random background music.
        self.background_channel.stop()
        self._queue_random_bgm()

        # Clear camera multithread queues.
        if ARGS.camera:
            if ARGS.deafy_camera_index is not None:
                # Lock the queue while clearing its contents.
                while not self.deafy_queue.empty():
                    self.deafy_queue.get()
                while not self.deafy_features_q.empty():
                    self.deafy_features_q.get()
            if ARGS.cat_camera_index is not None:
                while not self.cat_queue.empty():
                    self.cat_queue.get()
                while not self.cat_features_q.empty():
                    self.cat_features_q.get()

        if DEBUG_LEVEL >= DEBUG_PRINT_ALL:
            print "Game Reset."

    def quit(self):
        # Set stop flag to stop.
        with self.stop_flag.get_lock():
            self.stop_flag.value = 1
        if ARGS.camera:
            if self.deafy_cam_on:
                self.deafy_process.terminate()
                # self.deafy_process.join()
            if self.cat_cam_on:
                self.cat_process.terminate()
                # self.cat_process.join()

        if DEBUG_LEVEL >= DEBUG_PRINT_ONLY_CRUCIAL:
            print("Game exiting.")
        # Sleeps for 5 seconds before quitting
        time.sleep(2)

    def init_cams(self, which_cam_idx, display_location=None):

        # gets a list of available cameras.
        self.clist = pygame.camera.list_cameras()
        print (self.clist)

        if not self.clist:
            raise ValueError("Sorry, no cameras detected.")

        if which_cam_idx >= len(self.clist):
            raise IndexError("The camera index %d is not within the list of detected camera indices." %(which_cam_idx))

        # Now check whether the list of cameras, camera_shot_raw, and camera_default_display_location is initialized
        # correctly. If they're not a list or if they do not exist, create a new list.

        if not hasattr(self, 'camera'):
            # Initialize the list
            self.camera = [None for _ in range(len(self.clist))]
            self.camera_shot_raw = [None for _ in range(len(self.clist))]
            self.camera_shot = [None for _ in range(len(self.clist))]
            self.camera_default_display_location = [None for _ in range(len(self.clist))]

        cam_id = self.clist[which_cam_idx]
        # creates the camera of the specified size and in RGB colorspace
        self.camera[which_cam_idx] = pygame.camera.Camera(cam_id, (CAMERA_INPUT_WIDTH, CAMERA_INPUT_HEIGHT), "RGB")

        # starts the camera
        self.camera[which_cam_idx].start()

        # create a image to capture to.  for performance purposes, you want the
        # bit depth to be the same as that of the display image.
        self.camera_shot_raw[which_cam_idx] = pygame.surface.Surface((CAMERA_INPUT_WIDTH, CAMERA_INPUT_HEIGHT), 0, self.display)
        if display_location is None:
            self.camera_default_display_location[which_cam_idx] = (BATTLE_SCREEN_WIDTH - CAMERA_DISPLAY_WIDTH,
                                                                   BATTLE_SCREEN_HEIGHT - CAMERA_DISPLAY_HEIGHT)
        else:
            self.camera_default_display_location[which_cam_idx] = display_location

    def get_camera_shot(self, which_cam_idx):
        """
        Used for calibration. Maybe I can replace it with async later but it's not necessary for now.
        :param which_cam_idx: the index of the camera.
        :return: None
        """
        if self.camera[which_cam_idx] is None:
            raise IndexError("Can't get camera shot. Camera index %d is not initialized correctly!" %which_cam_idx)
        else:
            self.camera_shot_raw[which_cam_idx] = self.camera[which_cam_idx].get_image(self.camera_shot_raw[which_cam_idx])
            self.camera_shot[which_cam_idx] = pygame.transform.scale(self.camera_shot_raw[which_cam_idx], CAMERA_DISPLAY_SIZE)

    def get_camera_shot_async(self, which_cam_idx, q):
        """
        Asynchrounous get camera shot function. Stores the camera shots in the queue q.
        :param which_cam_idx: the index of the camera.
        :param q: a Queue to store camera shots in.
        :return: camera_shot_raw (size: CAMERA_INPUT_SIZE), camera_shot (size: CAMERA_DISPLAY_SIZE)
        """
        if self.camera[which_cam_idx] is None:
            raise IndexError("Can't get camera shot. Camera index %d is not initialized correctly!" %which_cam_idx)
        else:
            camera_shot_raw = self.camera[which_cam_idx].get_image(self.camera_shot_raw[which_cam_idx])
            camera_shot = pygame.transform.scale(self.camera_shot_raw[which_cam_idx], CAMERA_DISPLAY_SIZE)
            # Although it should not happen, this step prevents queue from getting too large.
            if q.qsize() < 5:
                q.put([pygame.image.tostring(camera_shot_raw, 'RGB'), pygame.image.tostring(camera_shot, 'RGB')])
            # I need to return the two because the facial feature extractor in the same thread needs to use them.
            return camera_shot_raw, camera_shot

    def blit_camera_shot(self, blit_location, which_cam_idx, q):
        """

        :param blit_location: tuple with format (x, y)
        :return:
        """
        # Get image when there is one. Otherwise it will wait for q to be filled and the fps will be limited by
        # child process.
        if q.qsize() > 0:
            camera_shot_pair = q.get()
            if camera_shot_pair is not None:
                # Get the second one - the resized camera shot.
                self.camera_shot_raw[which_cam_idx] = pygame.image.fromstring(camera_shot_pair[0],
                                                                          CAMERA_INPUT_SIZE , "RGB")
                self.camera_shot[which_cam_idx] = pygame.image.fromstring(camera_shot_pair[1],
                                                                          CAMERA_DISPLAY_SIZE , "RGB")
        if self.camera_shot[which_cam_idx] is None:
            raise IndexError("Can't blit camera shot. Camera index %d is not initialized correctly!" %which_cam_idx)
        else:
            self.display.blit(self.camera_shot[which_cam_idx], blit_location)


    def blit_photos(self, blit_location, photo_list, blit_location_delta):
        for i, (photo, photo_resized) in enumerate(photo_list):
            current_blit_location = (blit_location[0] + blit_location_delta[0] * i,
                                     blit_location[1] + blit_location_delta[1] * i)
            self.display.blit(photo_resized, current_blit_location)

    def _get_direction_from_keys(self, keys, wasd_constant_list):
        """
        This is for controlling the movement of cat or deafy using keyboard.
        :param keys: from pygame.keys.get_pressed()
        :param wasd_constant_list: a list containing constants for [UP, DOWN, LEFT, RIGHT]. So it would be
        [K_UP, K_DOWN, K_LEFT, K_RIGHT] if using the arrow keys.
        :return: (bool - true if the object should move ,a float from 0~2pi representing the direction)
        """
        assert len(wasd_constant_list) == 4
        _up, _down, _left, _right = wasd_constant_list
        dx = 0
        dy = 0
        if keys[_up]:
            dy += 1
        if keys[_down]:
            dy -= 1
        # In pygame display left and right is reversed from what we usually perceive.
        if keys[_left]:
            dx += 1
        if keys[_right]:
            dx -= 1

        if dx == 0 and dy == 0:
            return (False, 0)
        else:
            direction = np.arctan2(dy, dx) + math.pi  # [0, 2pi]
            return (True, direction)

    def _calibrate_camera(self, which_cam_idx, fld):
        while not fld.calibrate_face(self.camera_shot_raw[which_cam_idx], self.hpe, self.display):
            self.get_camera_shot(which_cam_idx)
            # self.blit_camera_shot((0, 0), which_cam_idx)
            # pygame.display.flip()

    def _get_facial_features_async(self, which_cam_idx, fld, camera_q, features_q):
        # enable multithreading in OpenCV for child thread
        cv2.setNumThreads(-1)
        while self.stop_flag.value == 0:
            camera_shot_raw, camera_shot = self.get_camera_shot_async(which_cam_idx, camera_q)
            # Now use the facial landmark defector
            # This step decreases the frame rate from 30 fps to 6fps. So we need to do something about it.
            # The speed is directly related to FACIAL_LANDMARK_PREDICTOR_WIDTH.
            face_coordinates_list, facial_features_list = fld.get_features(camera_shot_raw)
            if len(face_coordinates_list) > 0:
                if DEBUG_LEVEL >= DEBUG_PRINT_ALL:
                    print("Detected %d face%s in camera %d."
                          % (len(face_coordinates_list), "s" if len(face_coordinates_list) > 1 else "", which_cam_idx))
                # Assume the largest face is the target.
                face_index = fld.get_largest_face_index(face_coordinates_list)

                # Use head pose estimator
                head_pose = self.hpe.head_pose_estimation(facial_features_list[face_index])
                # Get the rotation invariant facial features.
                facial_features_3d = self.hpe.facial_features_to_3d(facial_features_list[face_index],
                                                                    head_pose[0], head_pose[1])
                features_q.put([face_coordinates_list, facial_features_list, face_index, head_pose, facial_features_3d])

    def _get_facial_scores(self, which_cam_idx, obj, fld, face_illustration, features_q):

        if features_q.qsize() > 0:
            features = features_q.get()
            if features is not None:
                face_coordinates_list, facial_features_list, face_index, head_pose, facial_features_3d = features
                # Update face illustration
                face_illustration.refresh_features(face_coordinates_list[face_index], facial_features_list[face_index],
                                                   head_pose[2])

                # Estimate pose difference from facing forward.
                pose_diff = fld.get_pose_diff(head_pose[0])
                if DEBUG_LEVEL >= DEBUG_PRINT_ALL:
                    print("Pose difference: %s" %(str(pose_diff)))
                # We only care about yaw (left right) and pitch (up down)

                # TODO: for now, don't vary the speed with respect to the pose. It's either constant speed moving or
                # stationary.
                is_moving, direction = get_direction_from_line(head_pose[2][0])
                if DEBUG_LEVEL >= DEBUG_PRINT_ALL:
                    print("Face %d direction %s" %(which_cam_idx, str(direction)))
                if is_moving:
                    obj.set_direction(direction)
                    obj.start_moving()
                else:
                    obj.stop_moving()

                mouth_open_score = get_mouth_open_score(facial_features_3d)

                # TODO: implement bullet CD aka recharge in deafy and cat. Replace deafy_bullet_need_recharge.
                if mouth_open_score >= MOUTH_SCORE_SHOOT_THRESHOLD:
                    bullet = obj.emit_bullets("BOUNCE", recharge=False)
                    if bullet:
                        self.bullets.append(bullet)
                elif mouth_open_score <= MOUTH_SCORE_RECHARGE_THRESHOLD:
                    obj.recharge_bullet("BOUNCE")

                if DEBUG_LEVEL >= DEBUG_PRINT_ALL:
                    print("Mouth open score: %f" % (mouth_open_score))

                # # Use the eye aspect ratio (aka blink detection) to jump
                prev_blink_counter = obj.blink_counter
                obj.blink_counter = get_any_eye_blink_rounds(facial_features_3d, obj.blink_counter)

                if DEBUG_LEVEL >= DEBUG_PRINT_ALL:
                    print("Number of frames that one of the eyes is closed: %d" % (obj.blink_counter))
                # Fire the spread bullet when the eye has been closed for a while and now it is open.
                if prev_blink_counter >= EYE_AR_CONSEC_FRAMES and obj.blink_counter == 0:
                    # No recharge for now...
                    bullets = obj.emit_bullets("SPREAD", recharge=True)
                    if bullets:
                        self.bullets = self.bullets + bullets

                # Now do automatically taking pictures.
                photo_score =  get_take_photo_score(mouth_open_score, head_pose[2][0], facial_features_3d)
                if photo_score > obj.max_photo_score:
                    obj.max_photo_score = photo_score
                    if self.camera_shot[which_cam_idx] is None:
                        raise IndexError("Can't get camera shot. Camera shot %d is empty!"
                                         % which_cam_idx)
                    obj.photo = self.camera_shot_raw[which_cam_idx].subsurface(Rect(face_coordinates_list[face_index]))

                # blink_score = get_blink_score(facial_features_3d)
                # # check to see if the eye aspect ratio is below the blink
                # # threshold, and if so, increment the blink frame counter
                # if blink_score < EYE_AR_THRESH:
                #     self.blink_counter += 1
                #
                # # otherwise, the eye aspect ratio is not below the blink
                # # threshold
                # else:
                #     # if the eyes were closed for a sufficient number of frames then jump proportional to the
                #     # number of frames that the eyes are closed.
                #     if self.blink_counter >= EYE_AR_CONSEC_FRAMES:
                #         self.deafy.jump(min(self.blink_counter * BLINK_JUMP_SPEED_FACTOR, MAX_JUMP_SPEED))
                #         self.blink_counter = 0

    def _queue_random_bgm(self):
        self.background_channel.queue(self.background_sounds[random.randint(0, len(self.background_sounds) - 1)])

    def _add_photo_to_list(self, photo_list, photo):
        photo_list.append((photo, pygame.transform.scale(photo, PHOTO_DISPLAY_SIZE)))

    def main(self,):
        # Before anything else, calibrate camera.

        if ARGS.camera:
            if self.deafy_cam_on:
                self._calibrate_camera(ARGS.deafy_camera_index, self.deafy_fld)
            if self.cat_cam_on:
                self._calibrate_camera(ARGS.cat_camera_index, self.cat_fld)

        self.init_battle()

        # Now start the asynchronous camera process.
        if ARGS.camera:
            if self.deafy_cam_on:
                self.deafy_process = Process(target=self._get_facial_features_async,
                                             args=(ARGS.deafy_camera_index,  self.deafy_fld,
                                                   self.deafy_queue, self.deafy_features_q))
                self.deafy_process.start()
                # self._get_facial_features_async(ARGS.deafy_camera_index, self.deafy_fld, self.deafy_queue)
            if self.cat_cam_on:
                self.cat_process = Process(target=self._get_facial_features_async,
                                           args=(ARGS.cat_camera_index, self.cat_fld,
                                                 self.cat_queue, self.cat_features_q))
                self.cat_process.start()
                # self._get_facial_features_async(ARGS.cat_camera_index, self.cat_fld, self.cat_queue)


        going = True
        self.clock = pygame.time.Clock()
        while going:

            if self.deafy.hp <= 0 or self.cat.hp <= 0:
                if self.deafy.hp <= 0 and self.cat.hp <= 0:
                    if DEBUG_LEVEL >= DEBUG_PRINT_ONLY_CRUCIAL:
                        print('Draw!')
                elif self.cat.hp <= 0:
                    if DEBUG_LEVEL >= DEBUG_PRINT_ONLY_CRUCIAL:
                        print('Cat ran out of hp. Deafy wins!')
                else:
                    if DEBUG_LEVEL >= DEBUG_PRINT_ONLY_CRUCIAL:
                        print('Deafy ran out of hp. Cat wins!')

                if self.deafy.hp <= 0:
                    # Now save the automatically taken photos of the players
                    if ARGS.camera and self.deafy_cam_on and self.deafy.photo is not None:
                        self._add_photo_to_list(self.deafy_player_photos, self.deafy.photo)
                    else:
                        self._add_photo_to_list(self.deafy_player_photos, self.default_face_photo)
                if self.cat.hp <= 0:
                    if ARGS.camera and self.cat_cam_on and self.cat.photo is not None:
                        self._add_photo_to_list(self.cat_player_photos, self.cat.photo)
                    else:
                        self._add_photo_to_list(self.cat_player_photos, self.default_face_photo)
                self.reset_battle()

            events = pygame.event.get()
            for e in events:
                if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                    going = False
                if e.type == END_BGM_EVENT and e.code == 0:
                    if DEBUG_LEVEL >= DEBUG_PRINT_ALL:
                        print("BGM ended. Playing a new one")
                    self._queue_random_bgm()


            # Get all key pressed.
            keys = pygame.key.get_pressed()

            if keys[K_ESCAPE]:
                going = False

            # handle different keys
            deafy_is_moving, deafy_direction = self._get_direction_from_keys(keys, WSAD_KEY_CONSTANTS)
            cat_is_moving, cat_direction = self._get_direction_from_keys(keys, ARROW_KEY_CONSTANTS)
            if deafy_is_moving:
                self.deafy.start_moving()
                self.deafy.set_direction(deafy_direction)
            else:
                self.deafy.stop_moving()
            if cat_is_moving:
                self.cat.start_moving()
                self.cat.set_direction(cat_direction)
            else:
                self.cat.stop_moving()

            if keys[K_SPACE]:
                if self.dialog_space_recharged and self.dialog_frame < DIALOG_FRAME_COUNT:
                    self.dialog_frame += 1
                    self.dialog_space_recharged = False
                    self.dialog.dialog_index = self.dialog_frame
                    if self.dialog_frame >= DIALOG_FRAME_COUNT:
                        self.is_dialog_active = False
                        self.dialog.is_active = self.is_dialog_active
                        self.reset_battle()
            else:
                self.dialog_space_recharged = True
            if keys[K_z]:
                # emit normal bullet
                bullet = self.deafy.emit_bullets("NORMAL", recharge=True)   # this is to make keyboard input easier (no recharge)
                if bullet:
                    self.bullets.append(bullet)
            if keys[K_x]:
                # emit bounce bullet
                bullet = self.deafy.emit_bullets("BOUNCE", recharge=True)
                if bullet:
                    self.bullets.append(bullet)
            if keys[K_c]:
                bullets = self.deafy.emit_bullets("SPREAD", recharge=True)
                if bullets:
                    self.bullets = self.bullets + bullets
            if keys[K_j]:
                bullet = self.cat.emit_bullets("NORMAL", recharge=True)
                if bullet:
                    self.bullets.append(bullet)
            if keys[K_k]:
                bullet = self.cat.emit_bullets("BOUNCE", recharge=True)
                if bullet:
                    self.bullets.append(bullet)
            if keys[K_l]:
                bullets = self.cat.emit_bullets("SPREAD", recharge=True)
                if bullets:
                    self.bullets = self.bullets + bullets


            if ARGS.camera and not self.is_dialog_active:
                if self.deafy_cam_on:
                    self._get_facial_scores(ARGS.deafy_camera_index, self.deafy, self.deafy_fld, self.deafy_player_face,
                                            self.deafy_features_q)
                if self.cat_cam_on:
                    self._get_facial_scores(ARGS.cat_camera_index, self.cat, self.cat_fld, self.cat_player_face,
                                            self.cat_features_q)


            # Now go through the bullets and see whether one hits anything
            # DEBUG
            # print("Number of bullets : %d" %(len(self.bullets)))
            for bullet in self.bullets:
                items_hit = bullet.items_hit(self.player_group)
                # for item in items_hit:
                #     item.set_failed(True)
                for item in items_hit:
                    # Otherwise the side that didn't get hit wins.
                    if bullet.owner != item:
                        if item == self.deafy:
                            self.deafy.take_damage(bullet)
                        else:
                            self.cat.take_damage(bullet)
                        bullet.kill()
                        self.bullets.remove(bullet)
                        if DEBUG_LEVEL >= DEBUG_PRINT_ALL:
                            print('HP Now: Deafy(%d) - Cat(%d)' % (self.deafy.hp, self.cat.hp))

            # Only keep the not destroyed objects.
            self.bullets = [bullet for bullet in self.bullets if not bullet.destroyed]


            # Clear screen and set every pixel to black..
            # TODO: Otherwise things that are blit directly onto the screen (not a sprite) is not erazed correctly.
            # Is there a better solution?
            self.display.fill(BLACK)

            # clear/erase the last drawn sprites
            self.all.clear(self.display, self.background)
            self.all.update()

            # # # draw the scene
            # No longer needs pygame.display.update(dirty). That updates part of the screen. Since we need to update
            # all pixels on screen anyway, that won't improve our speed.
            self.background_group.draw(self.display)
            self.obstacle_group.draw(self.display)
            self.front_group.draw(self.display)

            # display dialog
            #if self.is_dialog_active:
                # TODO: minor detail but it might be better to keep one single dialog object instead of creating a
                # new object every time. So like self.dialog.update_frame(self.dialog_frame) or something.
                #self.dialog.dialog_index = self.dialog_frame
                # Now the display blit is handled manually. Add it to a group and use methods like above to make sure
                # it is drawn after everything else. The blinking is likely caused by this bug.
                #self.display.blit(self.dialog.image, (BATTLE_SCREEN_WIDTH - 320, BATTLE_SCREEN_HEIGHT - 120))

            # enable camera only after all dialog frames are shown
            if ARGS.camera and not self.is_dialog_active:
                if self.deafy_cam_on:
                    self.blit_camera_shot(self.camera_default_display_location[ARGS.deafy_camera_index],
                                          ARGS.deafy_camera_index, self.deafy_queue)
                if self.cat_cam_on:
                    self.blit_camera_shot(self.camera_default_display_location[ARGS.cat_camera_index],
                                          ARGS.cat_camera_index, self.cat_queue)

            self.blit_photos(PHOTO_DISPLAY_DEAFY_BOTTOMLEFT, self.deafy_player_photos,
                             PHOTO_DISPLAY_DEAFY_DELTA)
            self.blit_photos(PHOTO_DISPLAY_CAT_BOTTOMLEFT, self.cat_player_photos,
                             PHOTO_DISPLAY_CAT_DELTA)
            self.clock.tick(MAX_FPS)
            self.text.blit_text_bottom_left_corner_at("FPS: %.1f" % (self.clock.get_fps()),
                                                      FPS_BLIT_BOTTOM_LEFT, self.display)
            pygame.display.flip()



def main():
    # Opencv hangs in multithreading. Solution is found here: https://github.com/opencv/opencv/issues/5150
    # disable multithreading in OpenCV for main thread to avoid problems after fork
    cv2.setNumThreads(0)

    pygame.init()
    pygame.camera.init()

    main_screen = MainScreen()
    try:
        main_screen.main()
    finally:
        # Otherwise threads won't quit even after the parent thread is already gone.
        main_screen.quit()
    pygame.quit()

if __name__ == '__main__':
    main()
