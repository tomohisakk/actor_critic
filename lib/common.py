import sys
import time
import numpy as np

GAMES = 30000
EPOCHES = 300

class RewardTracker:
	def __init__(self, writer):
		self.writer = writer

	def __enter__(self):
		self.ts = time.time()
		self.ts_frame = 0
		self.total_rewards = []
		self.total_n_steps_ep = []
		return self

	def __exit__(self, *args):
		self.writer.close()

	def reward(self, reward, frame, n_games):
		self.total_rewards.append(reward)
		n_steps_ep = frame - self.ts_frame
		self.total_n_steps_ep.append(n_steps_ep)
		n_epoches = int(len(self.total_rewards)/GAMES)
		speed = (frame - self.ts_frame) / (time.time() - self.ts)
		self.ts_frame = frame
		self.ts = time.time()
		mean_reward = np.mean(self.total_rewards[-GAMES:])
		mean_n_steps = np.mean(self.total_n_steps_ep[-GAMES:])
		if len(self.total_rewards) % 1000 == 0:
			print("epoches/games %d/%d, avg steps %d, mean reward %.3f, speed %.2f"
				%(n_epoches, n_games, mean_n_steps, mean_reward, speed))
			sys.stdout.flush()
		self.writer.add_scalar("speed", speed, n_games)
		self.writer.add_scalar("avg reward", mean_reward, n_games)
		self.writer.add_scalar("reward", reward, n_games)
		self.writer.add_scalar("avg steps", n_steps_ep, n_games)
		if n_epoches == EPOCHES:
			print("Finish %d epoches and %d games" % (n_epoches, n_games))
			return True
		return False