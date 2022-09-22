import torch as T
from train import AtariA2C
import ptan
from sub_envs.static import MEDAEnv
#from envs.dynamic import MEDAEnv
import numpy as np
import collections


from sub_envs.map import MakeMap
from sub_envs.map import Symbols

#from tensorboardX import SummaryWriter

def _is_touching(dstate, obj, map, dsize):
		i = 0
		while True:
			j = 0
			while True:
				if map[dstate[1]+j][dstate[0]+i] == obj:
					return True
				j += 1
				if j == dsize:
					break
			i += 1
			if i == dsize:
				break

		return False

def _compute_shortest_route(w, h, dsize, symbols,map, start):
	queue = collections.deque([[start]])
	seen = set([start])
#		print(self.map)
	while queue:
		path = queue.popleft()
#			print(path)
		x, y = path[-1]
		if _is_touching((x,y), symbols.Goal, map, dsize):
			return path
		for x2, y2 in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
			if 0 <= x2 < (w-dsize+1) and 0 <= y2 < (h-dsize+1) and \
			(_is_touching((x2,y2), symbols.Dynamic_module, map, dsize) == False) and\
			(_is_touching((x2,y2), symbols.Static_module, map, dsize) == False) and\
			(x2, y2) not in seen:
				queue.append(path + [(x2, y2)])
				seen.add((x2, y2))
#		print("Bad map")
#		print(self.map)
	return False


if __name__ == "__main__":
	###### Set params ##########
	ENV_NAME = "LR=0.0001_EB=0.001"
	TOTAL_GAMES = 1000

	W = 8
	H = 8
	DSIZE = 2
	P = 0.8

	############################
	env = MEDAEnv(w=W, h=H, dsize=DSIZE, p=P, test_flag=True)

	device = T.device('cpu')

#	writer = SummaryWriter(comment = "Result of " + ENV_NAME)
	CHECKPOINT_PATH = "saves/" + ENV_NAME

	net = AtariA2C(env.observation_space, env.action_space).to(device)
	net.load_checkpoint(CHECKPOINT_PATH)

	n_games = 0

	n_critical = 0

	map_symbols = Symbols()
	mapclass = MakeMap(w=W,h=H,dsize=DSIZE,p=P)

	while n_games != TOTAL_GAMES:
		done = False
		score = 0
		n_steps = 0
		map = mapclass.gen_random_map()
		observation = env.reset(test_map=map)

		path = _compute_shortest_route(W, H, DSIZE, map_symbols, map, (0,0))

		while not done:
			observation = T.tensor([observation], dtype=T.float).to(device)
			probs, _ = net(observation)
			action = T.argmax(probs).item()
		
			observation_, reward, done, message = env.step(action)

			score += reward
			observation = observation_

			if message == None:
				n_steps += 1

#		print("shortest:",len(path))
#		print("stepnum:",n_steps)

		if len(path) == n_steps:
			n_critical += 1

#		writer.add_scalar("Step_num", n_steps, n_games)
		n_games += 1

	print("Finish " + str(TOTAL_GAMES) + " tests")
	print("Num of critical path is ", n_critical)
	print("Avg of critical path is ", n_critical/TOTAL_GAMES)

