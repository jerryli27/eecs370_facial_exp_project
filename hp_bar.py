from background_objects import *

class HPBar(pygame.sprite.Sprite):
	""" This class represents the HP bar. """
	_HP_BAR_FULL_LENGTH = 100
	_HP_BAR_HEIGHT = 20
	_HP_COLOR = RED
	_BG_COLOR = WHITE

	def __init__(self, hp_max=PLAYER_HP, pos=GAME_SCREEN_RECT.bottomleft):
		pygame.sprite.Sprite.__init__(self, self.containers)
		self.pos = pos
		self.image = pygame.Surface([self._HP_BAR_FULL_LENGTH, self._HP_BAR_HEIGHT])
		self.image.fill(self._HP_COLOR)
		self.rect = self.image.get_rect(bottomleft=pos)
		self.hp = hp_max
		self.max_hp = float(hp_max)
		print 'created'

	def set_hp(self, hp_value):
		self.hp = hp_value

	def update(self):
		new_length = int(self._HP_BAR_FULL_LENGTH * (self.hp / self.max_hp))
		self.image.fill(self._HP_COLOR, rect=Rect(0, 0, new_length, self._HP_BAR_HEIGHT))
		self.image.fill(self._BG_COLOR, rect=Rect(new_length, 0, self._HP_BAR_FULL_LENGTH, self._HP_BAR_HEIGHT))
		self.rect = self.image.get_rect(bottomright=self.pos)