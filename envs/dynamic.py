import copy
import queue
import random
import numpy as np
from enum import IntEnum
import math
import collections
import gym

class Actions(IntEnum):
	N = 0
	E = 1
	S = 2
	W = 3

class Maps():
	State = "D"
	Goal = "G"
	Static_module = "#"
	Dynanic_module = "*"
	Health = "."

class MEDAEnv(gym.Env):
	def __init__(self, w=8, h=8, p=0.9, test_flag=False):
		super(MEDAEnv, self).__init__()
		assert w > 0 and h > 0
		assert 0 <= p <= 1.0
		self.w = w
		self.h = h
		self.p = p
		self.actions = Actions
		self.action_space = len(self.actions)
		self.observation_space = (w, h, 3)
		self.n_steps = 0
		self.max_step = 2*(w+h)

		self.state = (0,0)
		self.goal = (w-1, h-1)
		self.maps = Maps()
		self.map = self._gen_random_map()

		self.test_flag = test_flag

#		self.m_usage = np.zeros((l, w))

		self.dynamic_flag = 0
		self.dynamic_state = (0,0)

	def reset(self, test_map=None):
		self.n_steps = 0
		self.state = (0, 0)

		if self.test_flag == False:
			self.map = self._gen_random_map()
		else:
			self.map = test_map

#		self.m_usage = np.zeros((self.length, self.width))

		obs = self._get_obs()

		return obs

	def _is_map_good(self, map, start):
		queue = collections.deque([[start]])
		seen = set([start])
		while queue:
			path = queue.popleft()
#			print(path)
			x, y = path[-1]
			if map[y][x] == self.maps.Goal:
				return path
			for x2, y2 in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
				if 0 <= x2 < self.w and 0 <= y2 < self.h and \
				map[y2][x2] != self.maps.Dynanic_module and map[y2][x2] != self.maps.Static_module and (x2, y2) not in seen:
					queue.append(path + [(x2, y2)])
					seen.add((x2, y2))
		return False

	def _make_map(self):
		map = np.random.choice([".", "#", '*'], (self.h, self.w), p=[self.p, (1-self.p)/2, (1-self.p)/2])
		map[0][0] = "D"
		map[-1][-1] = "G"

		return map

	def _gen_random_map(self):
		map = self._make_map()
		while self._is_map_good(map, (0,0)) == False:
			map = self._make_map()
#		print(map)	
		return map

	def step(self, action):
		done = False
		message = None
		self.n_steps += 1

		_dist = self._get_dist(self.state, self.goal)

		self._update_position(action)

		if self.dynamic_flag == 1:
			dist = self._get_dist(self.dynamic_state, self.goal)
			self.dynamic_flag = 0
			message = "derror"
		else:
			dist = self._get_dist(self.state, self.goal)

		if dist == 0:
			reward = 1.0
			done = True
		elif self.n_steps == self.max_step:
			reward = -0.8
			done = True
		elif dist < _dist:
			reward = 0.5
		else:
			reward = -0.8
		
#		if self.test_flag == True:
#			print(self.map)

		obs = self._get_obs()
#		print(obs)
		return obs, reward, done, message

	def _get_dist(self, state1, state2):
		diff_x = state1[1] - state2[1]
		diff_y = state1[0] - state2[0]
		return math.sqrt(diff_x*diff_x + diff_y*diff_y)

	def _update_position(self, action):
		state_ = list(self.state)

		if action == Actions.N:
			state_[1] -= 1
		elif action == Actions.E:
			state_[0] += 1
		elif action == Actions.S:
			state_[1] += 1
		else:
			state_[0] -= 1

		if 0 <= state_[1] < self.w and 0 <= state_[0] < self.h and \
		   (self.map[state_[1]][state_[0]] == self.maps.Health or self.map[state_[1]][state_[0]] == self.maps.Goal):
#			self.m_usage[state_[1]][state_[0]] += 1
			self.map[self.state[1]][self.state[0]] = self.maps.Health
#			print(self.map)
			self.state = state_
			self.map[self.state[1]][self.state[0]] = self.maps.State
#			print(self.map)

		elif 0 <= state_[1] < self.w and 0 <= state_[0] < self.h and \
			 self.map[state_[1]][state_[0]] == self.maps.Dynanic_module:
#			print("Derror")
			self.dynamic_flag += 1
			self.dynamic_state = state_
			self.map[state_[1]][state_[0]] = self.maps.Static_module

	def _get_obs(self):
		obs = np.zeros(shape = (self.w, self.h, 3))
		for i in range(self.w):
			for j in range(self.h):
				if self.map[j][i] == self.maps.State:
					obs[j][i][0] = 1
				elif self.map[j][i] == self.maps.Goal:
					obs[j][i][1] = 1
				elif self.map[j][i] == self.maps.Static_module:
					obs[j][i][2] = 1
#		print(obs)
		return obs


	def close(self):
		pass
