import sys
import time
import numpy as np


class RewardTracker:
	def __init__(self, writer, stop_reward):
		self.writer = writer
		self.stop_reward = stop_reward

	def __enter__(self):
		self.ts = time.time()
		self.ts_frame = 0
		self.total_rewards = []
		self.total_n_steps_ep = []
		return self

	def __exit__(self, *args):
		self.writer.close()

	def reward(self, reward, frame):
		self.total_rewards.append(reward)
		n_steps_ep = frame - self.ts_frame
		self.total_n_steps_ep.append(n_steps_ep)
		n_epoches = int(len(self.total_rewards)/10000)
		n_games = int(len(self.total_rewards))
		speed = (frame - self.ts_frame) / (time.time() - self.ts)
		self.ts_frame = frame
		self.ts = time.time()
		mean_reward = np.mean(self.total_rewards[-100:])
		mean_n_steps = np.mean(self.total_n_steps_ep[-100:])
		if len(self.total_rewards) % 100 == 0:
			print("%d epoches, %d games, avg steps %d, mean reward %.3f, speed %.2f"
				%(n_epoches, n_games, mean_n_steps, mean_reward, speed))
			sys.stdout.flush()
		self.writer.add_scalar("speed", speed, frame)
		self.writer.add_scalar("reward_100", mean_reward, frame)
		self.writer.add_scalar("reward", reward, frame)
		self.writer.add_scalar("steps_100", n_steps_ep, frame)
		if n_epoches > 500:
			print("Finish %d epoches and %d games" % n_epoches, n_games)
			return True
		return False