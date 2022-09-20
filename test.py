import torch as T
from train import AtariA2C
import ptan
from envs.static import MEDAEnv
#from envs.dynamic import MEDAEnv
import numpy as np
import collections

from tensorboardX import SummaryWriter

class Maps():
	State = "D"
	Goal = "G"
	Static_module = "#"
	Dynanic_module = "*"
	Health = "."


def _is_map_good(w, h, map, start):
	queue = collections.deque([[start]])
	seen = set([start])
	while queue:
		path = queue.popleft()
		x, y = path[-1]
		if map[y][x] == Maps.Goal:
			return path
		for x2, y2 in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
			if 0 <= x2 < w and 0 <= y2 < h and \
			map[y2][x2] != Maps.Dynanic_module and map[y2][x2] != Maps.Static_module and (x2, y2) not in seen:
				queue.append(path + [(x2, y2)])
				seen.add((x2, y2))
	return False

def _gen_random_map(w, h, p1):
	while True:
		map = np.random.choice([".", "#", '*'], (w, h), p=[p1, (1-p1)/2, (1-p1)/2])
		map[0][0] = "D"
		map[-1][-1] = "G"
		if _is_map_good(w, h, map, (0,0)):
			break

	return map
"""
	print("============================================================================================")
	print("--- Start map ---")
	print(map)
	print("============================================================================================")
	print()
"""



if __name__ == "__main__":
	###### Set params ##########
	ENV_NAME = "LR=0.0001_EB=0.001"
	TOTAL_GAMES = 1000

	W = 8
	H = 8
	P = 0.8

	############################
	env = MEDAEnv(test_flag=True)

	device = T.device('cpu')

	writer = SummaryWriter(comment = "Result of " + ENV_NAME)
	CHECKPOINT_PATH = "saves/" + ENV_NAME

	net = AtariA2C(env.observation_space, env.action_space).to(device)
	net.load_checkpoint(CHECKPOINT_PATH)

	n_games = 0

	counter14 = 0
	counter32 = 0

	while n_games != TOTAL_GAMES:
		done = False
		score = 0
		n_steps = 0
		map = _gen_random_map(W, H, P)
		observation = env.reset(test_map=map)

		while not done:
			observation = T.tensor([observation], dtype=T.float).to(device)
			probs, _ = net(observation)
			action = T.argmax(probs).item()
		
			observation_, reward, done, _ = env.step(action)

			score += reward
			observation = observation_
			n_steps += 1

		if n_steps == 14:
			counter14 += 1
		elif n_steps == 32:
			counter32 += 1
		else:
			print("other: ", n_steps)

		writer.add_scalar("Step_num", n_steps, n_games)
		n_games += 1

	print("14 is ", counter14)
	print("32 is ", counter32)
	print("Finish " + str(TOTAL_GAMES) + " tests")