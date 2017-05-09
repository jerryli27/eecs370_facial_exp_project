from constants import *
from background_objects import *

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