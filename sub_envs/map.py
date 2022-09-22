import copy
import queue
import random
import numpy as np
from enum import IntEnum
import math
import collections
import gym

class Symbols():
	State = "D"
	Goal = "G"
	Static_module = "#"
	Dynamic_module = "*"
	Health = "."

class MakeMap():
	def __init__(self, w, h, dsize, p):
		super(MakeMap, self).__init__()
		assert w>0 and h>0 and dsize>0
		assert 0<=p<=1.0
		self.w = w
		self.h = h
		self.dsize = dsize
		self.p = p

		self.symbols = Symbols()
		self.map = self._make_map()

	def _make_map(self):
		map = np.random.choice([".", "#", '*'], (self.h, self.w), p=[self.p, (1-self.p)/2, (1-self.p)/2])
		i = 0

		# Set droplet
		while True:
			j = 0
			while True:
				map[j][i] = "D"
				j += 1
				if j == self.dsize:
					break
			i += 1
			if i == self.dsize:
				break

		# Set around the goal
		i = 0
		while True:
			j = 0
			while True:
				map[self.h-i-1][self.w-j-1] = "."
				j += 1
				if j == self.dsize:
					break
			i += 1
			if i == self.dsize:
				break

		map[-1][-1] = "G"

		self.map = map
#		return map


	def _is_touching(self, dstate, obj):
		i = 0
		while True:
			j = 0
			while True:
				if self.map[dstate[1]+j][dstate[0]+i] == obj:
					return True
				j += 1
				if j == self.dsize:
					break
			i += 1
			if i == self.dsize:
				break

		return False

	def _is_map_good(self, start):
		queue = collections.deque([[start]])
		seen = set([start])
#		print(self.map)
		while queue:
			path = queue.popleft()
#			print(path)
			x, y = path[-1]
			if self._is_touching((x,y), self.symbols.Goal):
				return True
			for x2, y2 in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
				if 0 <= x2 < (self.w-self.dsize+1) and 0 <= y2 < (self.h-self.dsize+1) and \
				(self._is_touching((x2,y2), self.symbols.Dynamic_module) == False) and\
				(self._is_touching((x2,y2), self.symbols.Static_module) == False) and\
				(x2, y2) not in seen:
					queue.append(path + [(x2, y2)])
					seen.add((x2, y2))
#		print("Bad map")
#		print(self.map)
		return False

	def gen_random_map(self):
		self._make_map()
		while self._is_map_good((0,0)) == False:
			self._make_map()
		return self.map
"""
if __name__ == '__main__':
	mapclass = MakeMap(w=10,h=10,dsize=3,p=0.7)
	map = mapclass.gen_random_map()
	print(map)
"""