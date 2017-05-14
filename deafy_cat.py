import math

from constants import *
from background_objects import *
from bullet import *
from hp_bar import *


class Deafy(pygame.sprite.Sprite):
    images = []
    sounds = []
    _STAND_IMAGE_INDEX = 0
    _LIE_DOWN_IMAGE_INDEX = 1
    _RUN_IMAGE_START_INDEX = 2
    _RUN_IMAGE_END_INDEX = 3
    _FAIL_INDEX = 4
    _JUMP_SOUND_INDEX = 1
    _VICTORY_SOUND_INDEX = 2

    def __init__(self, pos=BATTLE_SCREEN_RECT.bottomright):
        # Notice that bottomright instead of bottomleft is used for deafy, because deafy is facing right.
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.current_image_index = self._RUN_IMAGE_START_INDEX
        self.image = self.images[self.current_image_index]
        self.rect = self.image.get_rect(bottomright=pos)
        self.speed = 0  # The current speed
        self.moving_speed = DEFAULT_SPEED  # The speed IF deafy moves.
        self.direction = 0
        self.ground_level = pos[1]
        self.is_running = (INITIAL_DX < 0)
        self.gravity = INITIAL_GRAVITY
        self.failed = False  # If true, disable user control.
        self.hp = PLAYER_HP
        self.ap = PLAYER_AP
        self.hp_bar = self.create_hp_bar()

    # def move(self, pos):
    #     self.rect = self.image.get_rect(bottomright=pos)
    #     self.rect = self.rect.clamp(BATTLE_SCREEN_RECT)

    def set_direction(self, direction):
        """
        :param direction: a float between 0 and 2pi.
        :return: None
        """
        if not (direction <= math.pi * 2 and direction >= 0):
            raise AttributeError("Direction must be between 0 and 2pi. It is now: %f" %(direction))
        self.direction = direction

    def set_speed(self, new_speed):
        self.speed = new_speed

    def start_moving(self):
        self.speed = self.moving_speed

    def stop_moving(self):
        self.speed = 0

    def update(self):
        dx = self.speed * math.cos(self.direction)
        dy = self.speed * math.sin(self.direction)
        self.rect.move_ip(dx, dy)
        # Check if out of screen. If so, move back into the screen
        if BATTLE_SCREEN_RECT.colliderect(self.rect):
            self.rect.clamp_ip(BATTLE_SCREEN_RECT)

        if self.is_running:
            self.run_next_frame()

    def take_damage(self, bullet):
        self.hp -= bullet.ap
        self.hp_bar.set_hp(self.hp)

    def change_image(self, new_image_index):
        if self.current_image_index != new_image_index:
            self.current_image_index = new_image_index
            self.image = self.images[self.current_image_index]
            self.rect = self.image.get_rect(bottomright=self.rect.bottomright)

    def play_sound(self, sound_index):
        if sound_index >= len(self.sounds):
            raise IndexError("Sound index %d exceeding number of sounds stored (%d)." % (sound_index, len(self.sounds)))
        self.sounds[sound_index].play()

    def start_running(self):
        self.is_running = True
        self.change_image(self._RUN_IMAGE_START_INDEX)

    def stop_running(self):
        self.is_running = False
        self.change_image(self._STAND_IMAGE_INDEX)

    def run_next_frame(self):
        new_image_index = self.current_image_index + 1
        if new_image_index > self._RUN_IMAGE_END_INDEX:
            new_image_index = self._RUN_IMAGE_START_INDEX
        self.change_image(new_image_index)

    def set_failed(self, failed):
        if failed:
            self.failed = True
            self.is_running = False
            self.change_image(self._FAIL_INDEX)

    def emit_bullets(self, bullet_type, object_orientation, bullet_speed=BULLET_SPEED, bullet_color=BLACK):
        if bullet_speed <= 0:
            raise AttributeError("Bullet speed must be positive.")
        if object_orientation == "LEFT":
            bullet_location = self.rect.midleft
            # Move the bullet a little to avoid hiting the object firing the bullet.
            bullet_location = (bullet_location[0] - BULLET_SIZE, bullet_location[1])
            bullet_speed = -bullet_speed
        else:
            bullet_location = self.rect.midright
            # Move the bullet a little to avoid hiting the object firing the bullet.
            bullet_location = (bullet_location[0] + BULLET_SIZE, bullet_location[1])

        if bullet_type == "NORMAL":
            return NormalBullet(dx=bullet_speed, pos=bullet_location, ap=self.ap, color=bullet_color,
                                bullet_size=BULLET_SIZE)
        if bullet_type == "BOUNCE":
            return BounceBullet(dx=bullet_speed, pos=bullet_location, ap=self.ap, color=bullet_color,
                                bullet_size=BULLET_SIZE)
        if bullet_type == "SPREAD":
            spread_bullets = SpreadBullets(object_orientation, bullet_speed, ap=self.ap, pos=bullet_location,
                                           color=bullet_color, bullet_size=BULLET_SIZE)
            bullets_list = spread_bullets.create_bullets()
            return bullets_list

    def create_hp_bar(self):
        return HPBar(hp_max=self.hp, pos=HP_BAR_DEAFY_BOTTOMLEFT, name='DEAFY')

    def kill(self):
        self.hp_bar.kill()
        pygame.sprite.Sprite.kill(self)


class CatOpponent(Deafy):
    def create_hp_bar(self):
        return HPBar(hp_max=self.hp, pos=HP_BAR_CAT_BOTTOMLEFT, name='KITTY')


class CatObstacle(BackgroundObjects):
    _CAT_SIT_IMAGE_INDEX = 0
    _CAT_RUN_IMAGE_START_INDEX = 1
    _CAT_RUN_IMAGE_END_INDEX = 4

    def __init__(self, dx=STD_DX, pos=BATTLE_SCREEN_RECT.bottomleft, destroy_when_oos=True):
        super(CatObstacle, self).__init__(dx, pos, destroy_when_oos)
        if len(self.images) < (self._CAT_RUN_IMAGE_END_INDEX + 1):
            raise AssertionError("Wrong number of images loaded for class CatObstacle. "
                                 "It should be more than %d but it is now %d"
                                 % ((self._CAT_RUN_IMAGE_END_INDEX + 1), len(self.images)))

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
