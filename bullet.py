from constants import *
from background_objects import *

class Bullet(BackgroundObjects):
    """ This class represents the bullet . """

    def __init__(self, dx=INITIAL_DX, pos=SCREEN_RECT.bottomleft, destroy_when_oos=True, color=BLACK, bullet_size=BULLET_SIZE):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.image = pygame.Surface([bullet_size, bullet_size])
        self.image.fill(color)
        self.rect = self.image.get_rect(bottomleft=pos)
        self.dx = dx
        self.destroy_when_oos=destroy_when_oos
        self.destroyed = False

    def items_hit(self, block_list):
        # See if it hit a block
        block_hit_list = pygame.sprite.spritecollide(self, block_list, True)
        return block_hit_list