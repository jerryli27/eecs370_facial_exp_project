#!/usr/bin/env python

# This file builds a simple 2d arcade game.

import argparse
import os
import random
import time

import pygame
import pygame.camera
from pygame.locals import *

from facial_landmark_util import FacialLandmarkDetector, get_mouth_open_score, get_blink_score
from sprite_sheet import SpriteSheet

from constants import *
import dialog
from bullet import *
from obstacle import *
from deafy_cat import *


# Command line argument parser.
parser = argparse.ArgumentParser()

parser.add_argument("--no_camera", dest="camera", action="store_false",
                    help="Turn off camera and use keyboard control.")
parser.set_defaults(camera=True)

ARGS = parser.parse_args()



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

def fire_bullet(object_firing, object_orientation, bullet_speed=BULLET_SPEED, bullet_color=BLACK):
    """

    :param object_firing: The object firing the bullet. Must have the .rect attribute.
    :param object_orientation: "LEFT" or "RIGHT"
    :param bullet_speed: A positive number, it represents the speed of the bullet independent of its direction.
    :param bullet_color: A tuple representing its color.
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
            self.fld = FacialLandmarkDetector(SCREEN_WIDTH, SCREEN_HEIGHT, FACIAL_LANDMARK_PREDICTOR_WIDTH)

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
        CatOpponent.images = [cat_sheet.image_at((0, 0, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1, 158, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1+54, 158, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1+54*2, 158, 54, 42), colorkey=-1),
                              cat_sheet.image_at((1+54*3, 158, 54, 42), colorkey=-1), ]
        # Load sounds
        Deafy.sounds = [load_sound("normal.ogg"), load_sound("jump.ogg"), load_sound("victory.ogg")]
        CatOpponent.sounds=[dummysound(),dummysound(),dummysound()]  # TODO: add sound later.

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
        CatObstacle.containers = self.all, self.front_group
        CatOpponent.containers = self.all, self.front_group, self.player_group
        dialog.Dialog.containers = self.all
        Bullet.containers = self.all, self.front_group


    def init_battle(self):

        # # initialize stage
        # self.stage = Stage(num=1)
        # self.current_items = []

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
        self.cat = CatOpponent(pos=CAT_SCREEN_POS)
        self.ground_obstacle_sprites = []
        self.cat_obstacles = []
        self.bullets = []

        self.dx = INITIAL_DX
        self.visible_xrange = [0, SCREEN_WIDTH]

        # to track of which dialog frame shold be rendered
        self.dialog_frame = 0
        # To trach whether dialog is displayed now. If so, disable user control.
        self.is_dialog_active = False  # Disabled for demo purpose. Maybe add back later.

        # Facial feature detection things.
        self.blink_counter = 0
        self.deafy_bullet_need_recharge = False


    def reset_battle(self):
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
        self.is_dialog_active = False  # Disabled for demo purpose. Maybe add back later.

        # Facial feature detection things.
        self.blink_counter = 0
        self.deafy_bullet_need_recharge = False

        print "Game Reset."



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
        self.init_battle()
        going = True
        self.clock = pygame.time.Clock()
        while going:
            events = pygame.event.get()
            for e in events:
                if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                    going = False
                if e.type == KEYDOWN:
                    # handle different keys
                    if e.key == K_UP:
                        # Jump!
                        if self.cat.is_lying:
                            self.cat.stand_up()
                        else:
                            self.cat.jump()
                    if e.key ==K_LEFT:
                        self.bullets.append(fire_bullet(self.cat,"LEFT", bullet_color=RED))
                    if e.key == K_SPACE:
                        if self.dialog_frame < DIALOG_FRAME_COUNT:
                            self.dialog_frame += 1
                            if self.dialog_frame >= DIALOG_FRAME_COUNT:
                                self.is_dialog_active = False
                    if e.key == K_a:
                        # emit normal bullet
                        self.bullets.append(self.deafy.emit_bullets("NORMAL", "RIGHT", bullet_color=YELLOW))
                    if e.key == K_b:
                        # emit bounce bullet
                        self.bullets.append(self.deafy.emit_bullets("BOUNCE", "RIGHT", bullet_color=BLUE))
                    if e.key == K_s:
                        bullets = self.deafy.emit_bullets("SPREAD", "RIGHT", bullet_color=RED)
                        for bullet in bullets:
                            self.bullets.append(bullet)


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
                    if mouth_open_score >= MOUTH_SCORE_SHOOT_THRESHOLD and not self.deafy_bullet_need_recharge:
                        self.bullets.append(fire_bullet(self.deafy, "RIGHT", bullet_color=BLACK))
                        self.deafy_bullet_need_recharge = True
                    elif mouth_open_score <= MOUTH_SCORE_RECHARGE_THRESHOLD:
                        self.deafy_bullet_need_recharge = False

                    print("Mouth open degree: %f" %(mouth_open_score))

                    # Use the eye aspect ratio (aka blink detection) to jump
                    blink_score = get_blink_score(facial_features_list[0])
                    # check to see if the eye aspect ratio is below the blink
                    # threshold, and if so, increment the blink frame counter
                    if blink_score < EYE_AR_THRESH:
                        self.blink_counter += 1

                    # otherwise, the eye aspect ratio is not below the blink
                    # threshold
                    else:
                        # if the eyes were closed for a sufficient number of frames then jump proportional to the
                        # number of frames that the eyes are closed.
                        if self.blink_counter >= EYE_AR_CONSEC_FRAMES:
                            self.deafy.jump(min(self.blink_counter * BLINK_JUMP_SPEED_FACTOR,MAX_JUMP_SPEED))
                            self.blink_counter = 0



                else:
                    # TODO: maybe add a smoothing factor. Otherwise Deafy stops whenever the camera cannot detect the
                    # face, making the game harder to control.
                    self.set_dx(0)
                    self.deafy.set_gravity(INITIAL_GRAVITY)


            if self.deafy.y_speed <= 0 and self.deafy.rect.bottom > GROUND_LEVEL:
                self.deafy.land_on_ground(ground=GROUND_LEVEL)
            if self.cat.y_speed <= 0 and self.cat.rect.bottom > GROUND_LEVEL:
                self.cat.land_on_ground(ground=GROUND_LEVEL)

            # Now go through the bullets and see whether one hits anything
            # DEBUG
            # print("Number of bullets : %d" %(len(self.bullets)))
            for bullet in self.bullets:

                items_hit = bullet.items_hit(self.player_group)

                for item in items_hit:
                    item.set_failed(True)
                if len(items_hit) >= 2:
                    # If both got hit, draw
                    print('Draw!')
                    going = False
                elif len(items_hit) == 1:
                    # Otherwise the side that didn't get hit wins.
                    if items_hit[0] == self.deafy:
                        print('Deafy got hit. Cat wins!')
                        time.sleep(3)
                        self.reset_battle()
                    else:
                        print('Cat got hit. Deafy wins!')
                        time.sleep(3)
                        self.reset_battle()

            # Only keep the not destroyed objects.
            self.bullets = [bullet for bullet in self.bullets if not bullet.destroyed]
            # # based on dx, update the current screen state
            # self.visible_xrange = map(lambda n:n-self.dx, self.visible_xrange)
            # self.current_items = filter(lambda item:item.xstart + item.width >= self.visible_xrange[0], self.current_items)
            # if self.dx != 0:
            #     print self.visible_xrange
            # while self.stage.view_next_item() and self.stage.view_next_item().xstart < self.visible_xrange[1]:
            #     item = self.stage.pop_next_item()
            #     self.current_items.append(item)
            #     print self.visible_xrange, self.current_items
            #     # create ui object for this item
            #     xstart = item.xstart - self.visible_xrange[0]
            #     if item.type == 0:
            #         ystart = GROUND_LEVEL - item.height
            #     if item.type == 1:
            #         ystart = GROUND_LEVEL
            #     ystart += BACKGROUND_OBJECT_HEIGHT
            #     # how to show half sprites???
            #     self.ground_obstacle_sprites.extend([
            #         GroundObstacle(item_type=item.type, dx=self.dx, pos=(w, h))
            #         for w in range(xstart, xstart+item.width, BACKGROUND_OBJECT_WIDTH)
            #         for h in range(ystart, ystart+item.height, BACKGROUND_OBJECT_HEIGHT)])
            #
            #
            # # update the current game state
            # # update the ground level on the current screen
            # ground_level = [GROUND_LEVEL for i in xrange(SCREEN_WIDTH)]
            # for item in self.current_items:
            #     item_left = item.xstart - self.visible_xrange[0]
            #     for x in xrange(max(0, item_left), min(SCREEN_WIDTH, item_left+item.width)):
            #         if item.type == 0:
            #             ground_level[x] -= item.height
            #         elif item.type == 1:
            #             ground_level[x] += item.height
            # # if the ground level in front of deafy is higher than itself, stop running.
            # dleft, dright = self.deafy.rect.left, self.deafy.rect.right
            # dbottom, dy = self.deafy.rect.bottom, self.deafy.y_speed
            # for x in xrange(dright, dright-self.dx):
            #     if self.deafy.rect.bottom > ground_level[x]:
            #         self.set_dx(0)  # TODO: also disable set_dx() anywhere else.
            #         self.deafy.failed = True  # Disable user control on deafy afterwards.
            # # check the dog's motion with respect to the ground
            # valid_ground_level = min(ground_level[dleft:dright])    # use the highest ground level
            # if dbottom <= valid_ground_level <= dbottom-dy:
            #     if self.deafy.y_speed <= 0:
            #         self.deafy.land_on_ground(ground=valid_ground_level)
            # elif dbottom-dy < valid_ground_level:
            #     self.deafy.fall()

            # # check the status of the game (win / lose)
            # # winning: if the dog succeeded going 100 pixel further than the last stage item
            # # TODO: add a flag or a person or something to indicate end point, after the demo.
            # if self.stage.checkWin(self.deafy.rect.left+self.visible_xrange[0]):
            #     print 'You win!'
            #     self.deafy.play_sound(self.deafy._VICTORY_SOUND_INDEX)
            #     time.sleep(5)  # TODO: Sleep this amount of seconds to make sure sound finishes playing. Improve this part after the demo to add victory screen and fail screen.
            #     going = False
            # # lose if the dog falls completely outside the screen
            # if self.deafy.rect.top > SCREEN_HEIGHT:
            #     self.handle_game_over()
            #     going = False


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
                self.dialog = dialog.Dialog(self.dialog_frame)
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

        # Sleeps for 5 seconds before quitting
        time.sleep(5)



def main():
    pygame.init()
    pygame.camera.init()

    MainScreen().main()
    pygame.quit()

if __name__ == '__main__':
    main()
