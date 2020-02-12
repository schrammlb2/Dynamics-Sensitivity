import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class TransitionModel(nn.Module):
	def __init__(self, state_dim=4, action_dim=1, action_type='Categorical'):
		super(TransitionModel, self).__init__()
		h = 256
		self.action_type = action_type

		self.state_enc = torch.nn.Linear(state_dim, h)
		if action_type == 'Categorical':
			self.action_enc = torch.nn.Linear(1, h)
		else: 
			self.action_enc = torch.nn.Linear(action_dim, h)
		
		self.transition = torch.nn.Sequential(
			torch.nn.ReLU(),
			# torch.nn.Dropout(.1),
			torch.nn.Linear(h, h),
			torch.nn.ReLU(),
			# torch.nn.Dropout(.1),     
		)

		self.state_pred = torch.nn.Linear(h, state_dim)
		# self.reward_pred = torch.nn.Linear(h, 1)

	def forward(self, state, action):
		enc = self.state_enc(state) + self.action_enc(action)
		trans_enc = self.transition(enc)
		state_change = self.state_pred(trans_enc)
		return state
		# reward = self.reward_pred(trans_enc).squeeze(-1)
		# return (state + state_change, reward)