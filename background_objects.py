import pygame
from constants import *

class BackgroundObjects(pygame.sprite.Sprite):
    images = []
    def __init__(self, dx=INITIAL_DX, dy=INITIAL_DY, pos=BATTLE_SCREEN_RECT.bottomleft, destroy_when_oos=False):
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
        self.dy = dy
        self.destroy_when_oos=destroy_when_oos
        self.destroyed = False


    def update(self):
        # Makes the enemy move in the x direction.
        self.rect.move_ip(self.dx, self.dy)

        # If the enemy is outside of the platform, make it appear on the other side of the screen
        if self.rect.left > BATTLE_SCREEN_RECT.right or self.rect.right < BATTLE_SCREEN_RECT.left:
            if self.destroy_when_oos:
                self.handle_oos()
                return
            else:
                if self.rect.left > BATTLE_SCREEN_RECT.right:
                    # Move the sprite towards the left n pixels where n = width of platform + width of object.
                    self.rect.move_ip(BATTLE_SCREEN_RECT.left - BATTLE_SCREEN_RECT.right - BACKGROUND_OBJECT_WIDTH, 0)
                elif self.rect.right < BATTLE_SCREEN_RECT.left:
                    # Move the sprite towards the right n pixels where n = width of platform + width of object.
                    self.rect.move_ip(BATTLE_SCREEN_RECT.right - BATTLE_SCREEN_RECT.left + BACKGROUND_OBJECT_WIDTH, 0)
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
            self.destroyed = True
            self.kill()


class Ground(BackgroundObjects):
    pass

class Sky(BackgroundObjects):
    pass
