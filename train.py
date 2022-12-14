import os
import gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch as T
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

from lib import common

from sub_envs.static import MEDAEnv
#from sub_envs.dynamic import MEDAEnv

GAMMA = 0.99
LEARNING_RATE = 1e-4
ENTROPY_BETA = 1e-3
BATCH_SIZE = 64
W = 8
H = 8
DSIZE = 1
P = 0.9

USEGPU = False
OPTIMIZER= "Adam"	#Adam or SGD

REWARD_STEPS = 1
CLIP_GRAD = 0.1
SGAMMA = 0.9


class AtariA2C(nn.Module):
	def __init__(self, input_shape, n_actions):
		super(AtariA2C, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(input_shape[0], 64, 1, stride=1),
			nn.ReLU(),
			nn.Conv2d(64, 128, 2, stride=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, 2, stride=1),
			nn.ReLU()
		)

		conv_out_size = self._get_conv_out(input_shape)
		self.policy = nn.Sequential(
			nn.Linear(conv_out_size, 128),
			nn.ReLU(),
			nn.Linear(128, n_actions)
		)

		self.value = nn.Sequential(
			nn.Linear(conv_out_size, 128),
			nn.ReLU(),
			nn.Linear(128, 1)
		)

	def _get_conv_out(self, shape):
		o = self.conv(T.zeros(1, *shape))
		return int(np.prod(o.size()))

	def forward(self, x):
		fx = x.float()/2
		conv_out = self.conv(fx).view(fx.size()[0], -1)
		return self.policy(conv_out), self.value(conv_out)

	def save_checkpoint(self, checkpoint_path):
		print("... saveing checkpoint ...")
		T.save(self.state_dict(), checkpoint_path)

	def load_checkpoint(self, checkpoint_path):
		self.load_state_dict(T.load(checkpoint_path))


def unpack_batch(batch, net, device='cpu'):
	states = []
	actions = []
	rewards = []
	not_done_idx = []
	last_states = []
	for idx, exp in enumerate(batch):
		states.append(np.array(exp.state, copy=False))
		actions.append(int(exp.action))
		rewards.append(exp.reward)
		if exp.last_state is not None:
			not_done_idx.append(idx)
			last_states.append(np.array(exp.last_state, copy=False))

	states_v = T.FloatTensor(np.array(states, copy=False)).to(device)
	actions_t = T.LongTensor(actions).to(device)

	# handle rewards
	rewards_np = np.array(rewards, dtype=np.float32)
	if not_done_idx:
		last_states_v = T.FloatTensor(np.array(last_states, copy=False)).to(device)
		last_vals_v = net(last_states_v)[1]
		last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
		last_vals_np *= GAMMA ** REWARD_STEPS
		rewards_np[not_done_idx] += last_vals_np

#	print(rewards_np)
	ref_vals_v = T.FloatTensor(rewards_np).to(device)

	return states_v, actions_t, ref_vals_v


if __name__ == "__main__":

	env = MEDAEnv(w=W, h=H, dsize=DSIZE, p=P)
	env_name = "LR=" + str(LEARNING_RATE) + "_EB=" + str(ENTROPY_BETA)
	writer = SummaryWriter(comment = env_name)

	if not os.path.exists("saves"):
		os.makedirs("saves")
	checkpoint_path = "saves/" + env_name

	if USEGPU == True:
		device = T.device('cuda:0' if T.cuda.is_available else 'cpu')
	else:
		device = T.device('cpu')
	print("Device is ", device)

	net = AtariA2C(env.observation_space, env.action_space).to(device)
	print(net)

	agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
	exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)


	if OPTIMIZER == "Adam":
		optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
	elif OPTIMIZER == "SGD":
		optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
	else:
		print("Optimizer not found")

#	scheduler = T.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.sgamma)

	batch = []

	n_games = 0

	with common.RewardTracker(writer) as tracker:
		with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
			for step_idx, exp in enumerate(exp_source):
#				print(exp.reward)
				batch.append(exp)

				# handle new rewards
				new_rewards = exp_source.pop_total_rewards()
				if new_rewards:
					n_games += 1
					if n_games%30000 == 0:
						net.save_checkpoint(checkpoint_path)
#						scheduler.step()

					if tracker.reward(new_rewards[0], step_idx, n_games):
						break

				if len(batch) < BATCH_SIZE:
					continue

				states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
				batch.clear()

				optimizer.zero_grad()
				logits_v, value_v = net(states_v)
				loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

				log_prob_v = F.log_softmax(logits_v, dim=1)
				adv_v = vals_ref_v - value_v.detach()
				log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
				loss_policy_v = -log_prob_actions_v.mean()

				prob_v = F.softmax(logits_v, dim=1)
				entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

				# calculate policy gradients only
				loss_policy_v.backward(retain_graph=True)
				grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
										for p in net.parameters()
										if p.grad is not None])

				# apply entropy and value gradients
				loss_v = entropy_loss_v + loss_value_v
				loss_v.backward()
				nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
				optimizer.step()
				# get full loss
				loss_v += loss_policy_v
					
				tb_tracker.track("advantage",       adv_v, n_games)
				tb_tracker.track("values",          value_v, n_games)
				tb_tracker.track("batch_rewards",   vals_ref_v, n_games)
				tb_tracker.track("loss_entropy",    entropy_loss_v, n_games)
				tb_tracker.track("loss_policy",     loss_policy_v, n_games)
				tb_tracker.track("loss_value",      loss_value_v, n_games)
				tb_tracker.track("loss_total",      loss_v, n_games)
				tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), n_games)
				tb_tracker.track("grad_max",        np.max(np.abs(grads)), n_games)
				tb_tracker.track("grad_var",        np.var(grads), n_games)
