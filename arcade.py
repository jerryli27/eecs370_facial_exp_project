#!/usr/bin/env python

# This file builds a simple 2d arcade game.


from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import random
from PIL import Image

import pygame
import pygame.camera
from pygame.locals import *

from sprite_sheet import SpriteSheet

#game constants
SCREEN_HEIGHT=480
SCREEN_WIDTH=640
SCREEN_RECT= Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
BACKGROUND_OBJECT_HEIGHT = 32
BACKGROUND_OBJECT_WIDTH = 32
MAX_FPS = 30
GRAVITY = 2  # pixel/second^2
MAX_JUMP_CHARGE = 2  # The number of time the object can jump


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

        deafy_sheet = SpriteSheet("data/Undertale_Annoying_Dog.png")
        deafy_sheet_transparent_color = ()

        Deafy.images = [load_image('dog.gif', flip_x=True),
                        deafy_sheet.image_at((2, 204, 26-2, 216-204), colorkey=-1),
                        deafy_sheet.image_at((2, 182, 23-2, 200-182), colorkey=-1),
                        deafy_sheet.image_at((25, 182, 44-25, 200-182), colorkey=-1),]
        Sky.images =  [load_image('sky.png', (32,32))]
        Ground.images =  [load_image('grass.png', (32,32))]
        # Initialize Game Groups
        self.all = pygame.sprite.RenderUpdates()
        self.background_group = pygame.sprite.RenderUpdates()
        # Sprites in this group are rendered after background so that they appear on the top.
        self.front_group = pygame.sprite.RenderUpdates()

        # assign default groups to each sprite class
        Deafy.containers = self.all, self.front_group
        Ground.containers = self.all, self.background_group
        Sky.containers = self.all, self.background_group

        # The y goes from top to bottom starting at 0.
        self.ground_y_limits = (SCREEN_HEIGHT*3/4,SCREEN_HEIGHT)
        self.sky_y_limits = (0,SCREEN_HEIGHT*3/4)
        # TODO: Maybe the height and width are the other way around
        self.ground_sprites = [Ground(pos=(w*BACKGROUND_OBJECT_WIDTH, h*BACKGROUND_OBJECT_HEIGHT))
                               for w in range(SCREEN_WIDTH / BACKGROUND_OBJECT_WIDTH + 1)
                               for h in range(self.ground_y_limits[0] / BACKGROUND_OBJECT_HEIGHT,
                                              self.ground_y_limits[1] / BACKGROUND_OBJECT_HEIGHT)]
        self.sky_sprites = [Sky(pos=(w*BACKGROUND_OBJECT_WIDTH, h*BACKGROUND_OBJECT_HEIGHT))
                               for w in range(SCREEN_WIDTH / BACKGROUND_OBJECT_WIDTH + 1)
                               for h in range(self.sky_y_limits[0] / BACKGROUND_OBJECT_HEIGHT,
                                              self.sky_y_limits[1] / BACKGROUND_OBJECT_HEIGHT)]
        self.deafy = Deafy(pos=(SCREEN_WIDTH/8, SCREEN_HEIGHT*7/8))

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
                        # Decrease the speed for all background sprites, so it looks like deafy is moving to the right.
                        for s in self.ground_sprites + self.sky_sprites:
                            s.plus_dx(-1)
                    if e.key == K_LEFT:
                        # Increase the speed for all background sprites, so it looks like deafy is moving to the left.
                        for s in self.ground_sprites + self.sky_sprites:
                            s.plus_dx(1)
                    if e.key == K_UP:
                        # Jump!
                        if self.deafy.is_lying:
                            self.deafy.stand_up()
                        else:
                            self.deafy.jump()
                    if e.key == K_DOWN:
                        # Lie down.
                        self.deafy.lie_down()

            # clear/erase the last drawn sprites
            self.all.clear(self.display, self.background)
            self.all.update()
            # # draw the scene
            dirty = self.background_group.draw(self.display)
            pygame.display.update(dirty)
            dirty = self.front_group.draw(self.display)
            pygame.display.update(dirty)


            # dirty = self.all.draw(self.display)
            # pygame.display.update(dirty)
            pygame.display.flip()
            self.clock.tick(MAX_FPS)
            print (self.clock.get_fps())

class Deafy(pygame.sprite.Sprite):
    images = []
    def __init__(self, pos=SCREEN_RECT.midbottom):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.current_image_index = 0
        self.image = self.images[self.current_image_index]
        self.rect = self.image.get_rect(midbottom=pos)
        self.y_speed = 0
        self.ground_level = pos[1]
        self.is_jumping = False
        self.jump_charge = MAX_JUMP_CHARGE  # The number of time the object can jump
        self.is_lying = False

    def move(self, pos):
        self.rect= self.image.get_rect(midbottom=pos)
        self.rect = self.rect.clamp(SCREEN_RECT)

    def jump(self):
        if self.jump_charge > 0:
            self.is_jumping = True
            # if the object was falling too fast, make the second jump weaker but still allow it to jump.
            self.y_speed = max(10, self.y_speed + random.randrange(10,50))
            self.jump_charge -= 1

    def update(self):
        if self.is_jumping:
            self.y_speed -= GRAVITY
            self.rect.move_ip(0,-self.y_speed)
            # Because y goes from top to bottom.
            if self.rect.bottom > self.ground_level:
                self.rect.bottom = self.ground_level
                self.is_jumping = False
                self.jump_charge = MAX_JUMP_CHARGE

    def change_image(self, new_image_index):
        if self.current_image_index != new_image_index:
            self.current_image_index = new_image_index
            self.image = self.images[self.current_image_index]
            self.rect = self.image.get_rect(midbottom=self.rect.midbottom)

    def lie_down(self):
        if not self.is_lying and not self.is_jumping:
            self.is_lying = True
            self.change_image(1)

    def stand_up(self):
        if self.is_lying:
            self.is_lying = False
            self.change_image(0)




class BackgroundObjects(pygame.sprite.Sprite):
    images = []
    def __init__(self, pos=SCREEN_RECT.midbottom, destroy_when_oos=False):
        """

        :param pos: The initial position of the object.
        :param destroy_when_oos: If true, the object self destroys when it is out of the screen. If False, the object
        wraps around the screen when it is out of the screen.
        """
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.current_image_index = 0
        self.image = self.images[self.current_image_index]
        self.rect = self.image.get_rect(topleft=pos)
        self.dx = -1
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
            self.rect = self.image.get_rect(midbottom=self.rect.midbottom)


class Ground(BackgroundObjects):
    images = []


class Sky(BackgroundObjects):
    images = []



def main():
    pygame.init()
    pygame.camera.init()

    MainScreen().main()
    pygame.quit()

if __name__ == '__main__':
    main()
