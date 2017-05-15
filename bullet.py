from constants import *
from background_objects import *
import random

class Bullet(BackgroundObjects):
    """ This class represents the bullet . """

    def __init__(self, dx=INITIAL_DX, dy=INITIAL_DY, pos=BATTLE_SCREEN_RECT.bottomleft, ap=PLAYER_AP,
                 destroy_when_oos=True, color=BLACK, bullet_size=BULLET_SIZE):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = pygame.Surface([bullet_size, bullet_size])
        self.image.fill(color)
        self.rect = self.image.get_rect(bottomleft=pos)
        self.dx = dx
        self.dy = dy
        self.ap = ap
        self.destroy_when_oos=destroy_when_oos
        self.destroyed = False

    def items_hit(self, block_list):
        # See if it hit a block
        block_hit_list = pygame.sprite.spritecollide(self, block_list, False)
        return block_hit_list

class NormalBullet(Bullet):
    pass

class BounceBullet(Bullet):
    """ Bullet bounce back when hit the BATTLE_SCREEN_RECT """

    def update(self):
        # print "Bullet bounce back when hit the BATTLE_SCREEN_RECT", self.dy
        self.rect.move_ip(self.dx, self.dy)

        if self.rect.bottom >= BATTLE_SCREEN_RECT.bottom:
            self.handle_oos()
            return

        # if bullet hits the BATTLE_SCREEN_RECT, set the speed in oppositive direction
        if self.rect.right >= BATTLE_SCREEN_RECT.right or self.rect.left <= BATTLE_SCREEN_RECT.left:
            self.dy = random.randint(-BULLET_BOUNCE_H_SPEED, BULLET_BOUNCE_H_SPEED)
            self.dx = -self.dx
            print "Bounce Back!", self.dy

class SpreadBullets(Bullet):
    """ Bullets emits more bullets along the trajectory """

    def __init__(self, bullet_orientation, dx=INITIAL_DX, dy=INITIAL_DY, pos=BATTLE_SCREEN_RECT.bottomleft,
                 ap=PLAYER_AP, destroy_when_oos=True, color=BLACK, bullet_size=BULLET_SIZE):
        # Call the parent class (Sprite) constructor
        #pygame.sprite.Sprite.__init__(self, self.containers)
        self.bullet_orientation = bullet_orientation
        self.bullet_color = color
        self.bullets = []
        self.bullets.append(NormalBullet(dx, dy, pos=pos, ap=ap, color=color, bullet_size=bullet_size))

        self.dx = dx
        self.dy = 1
        self.ap = ap
        self.destroy_when_oos=destroy_when_oos
        self.destroyed = False
        self.num_frontmost = 2


        for i in range(self.num_frontmost):
            self.clone_bullets(i+1)


        # self.image = pygame.Surface([bullet_size, bullet_size])
        # self.image.fill(_HP_COLOR)
        # self.rect = self.image.get_rect(bottomleft=pos)


    def create_bullets(self):
        print "self.bullets", len(self.bullets)
        return self.bullets

    def update(self):
        for bullet in self.bullets:
            bullet.rect.move_ip(self.dx, self.dy)

            if not BATTLE_SCREEN_RECT.colliderect(self.rect):
                bullet.handle_oos()
                self.bullets.remove(bullet)
                return

    def items_hit(self, block_list):
        # See if it hit a block
        block_hit_list = []
        for bullet in self.bullets:
            block_hit_list + pygame.sprite.spritecollide(bullet, block_list, True)

        return block_hit_list


    def clone_bullets(self, num_frontmost):
        """
        add two more new bullets for every frontmost bullets
        """
        counter = 0
        clone = []
        offset = 10

        for bullet in reversed(self.bullets):
            if counter >= num_frontmost:
                break
            counter += 1

            if self.bullet_orientation == "LEFT":
                bullet_location = bullet.rect.midleft
                # clone two bullets, each offset from original bullet in y axis
                bullet_location1 = (bullet_location[0] - BULLET_SIZE, bullet_location[1] + offset)
                bullet_location2 = (bullet_location[0] - BULLET_SIZE, bullet_location[1] - offset)
            else:
                bullet_location = bullet.rect.midright
                bullet_location1 = (bullet_location[0] + BULLET_SIZE, bullet_location[1] + offset)
                bullet_location2 = (bullet_location[0] + BULLET_SIZE, bullet_location[1] - offset)

            clone.append(NormalBullet(self.dx, self.dy, pos=bullet_location1, color=self.bullet_color, bullet_size=BULLET_SIZE))
            clone.append(NormalBullet(self.dx, -self.dy, pos=bullet_location2, color=self.bullet_color, bullet_size=BULLET_SIZE))

        self.bullets = self.bullets + clone
