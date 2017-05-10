from constants import *
from background_objects import *
from bullet import *

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
    def __init__(self, pos=SCREEN_RECT.bottomright):
        # Notice that bottomright instead of bottomleft is used for deafy, because deafy is facing right.
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.current_image_index = self._RUN_IMAGE_START_INDEX
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
            if speed is None:
                d_jump_speed = 25
            else:
                d_jump_speed = speed

            self.y_speed = max(10, self.y_speed + d_jump_speed)
            self.jump_charge -= 1
            # Play sound effect
            self.play_sound(self._JUMP_SOUND_INDEX)
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
            self.change_image(self._LIE_DOWN_IMAGE_INDEX)

    def stand_up(self):
        if self.failed:
            return
        if self.is_lying:
            self.is_lying = False
            self.change_image(self._STAND_IMAGE_INDEX)

    def start_running(self):
        # if self.is_lying:
        #     print("Warning! Deafy attempted to start_running while it's lying down.")
        # else:
        #     self.is_running = True
        self.is_running = True
        self.change_image(self._RUN_IMAGE_START_INDEX)

    def stop_running(self):
        # Only stop animation when it's not jumping.
        if not self.is_jumping:
            self.is_running = False
            self.change_image(self._STAND_IMAGE_INDEX)

    def run_next_frame(self):
        new_image_index = self.current_image_index + 1
        if new_image_index > self._RUN_IMAGE_END_INDEX:
            new_image_index = self._RUN_IMAGE_START_INDEX
        self.change_image(new_image_index)

    def set_gravity(self, new_gravity):
        """
        Modifies self.gravity to be the new gravity. Gravity cannot be lower than 0.
        :param new_gravity:
        :return: Nothing
        """
        self.gravity = max(MIN_GRAVITY,new_gravity)

    def set_failed(self, failed):
        if failed:
            self.failed = True
            self.is_running = False
            self.is_jumping = False
            self.change_image(self._FAIL_INDEX)

    def emit_bullets(self, bullet_type, object_orientation, bullet_speed=BULLET_SPEED, bullet_color=BLACK):
        if bullet_speed <= 0:
            raise AttributeError("Bullet speed must be positive.")
        if object_orientation == "LEFT":
            bullet_location = self.rect.midleft
            # Move the bullet a little to avoid hiting the object firing the bullet.
            bullet_location = (bullet_location[0]-BULLET_SIZE, bullet_location[1])
            bullet_speed = -bullet_speed
        else:
            bullet_location = self.rect.midright
            # Move the bullet a little to avoid hiting the object firing the bullet.
            bullet_location = (bullet_location[0] + BULLET_SIZE, bullet_location[1])

        if bullet_type == "NORMAL":
            return NormalBullet(dx=bullet_speed, pos=bullet_location, color=bullet_color,bullet_size=BULLET_SIZE)
        if bullet_type == "BOUNCE":
            return BounceBullet(dx=bullet_speed, pos=bullet_location, color=bullet_color,bullet_size=BULLET_SIZE)
        if bullet_type == "SPREAD":
            spread_bullets = SpreadBullets(object_orientation, bullet_speed, pos=bullet_location, color=bullet_color, bullet_size=BULLET_SIZE)
            bullets_list = spread_bullets.create_bullets()
            return bullets_list


class CatOpponent(Deafy):
    pass



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
