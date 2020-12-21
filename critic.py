import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def weights_init_(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_dim):
		super(QNetwork, self).__init__()

		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = nn.Linear(hidden_dim, hidden_dim)
		self.linear4 = nn.Linear(hidden_dim, 1)
		self.apply(weights_init_)

	def forward(self, state, action):
		xu = torch.cat([state, action], 1)
		x1 = F.relu(self.linear1(xu))
		x1 = F.relu(self.linear2(x1))
		x1 = F.relu(self.linear3(x1))
		x1 = self.linear4(x1)

		return x1

	def loss(self, memory, batch_size, optimistic=False, state=None, action=None, eta=None):
		states, actions, qvalues=memory.sample(batch_size)
		states = torch.FloatTensor(states).to(device)
		actions = torch.FloatTensor(actions).to(device)
		qvalues = torch.FloatTensor(qvalues).to(device)
		predict_qvalues = self.forward(states, actions)
		mse_loss = F.mse_loss(predict_qvalues.reshape(-1), qvalues)
		if optimistic==False:
			return mse_loss
		if eta is None:
			eta=1.0

		reward_loss=self.forward(state, action)
		return -eta * reward_loss + mse_loss


class DoubleQNetwork(nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_dim):
		super(DoubleQNetwork, self).__init__()

		self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = nn.Linear(hidden_dim, 1)

		self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
		self.linear5 = nn.Linear(hidden_dim, hidden_dim)
		self.linear6 = nn.Linear(hidden_dim, 1)
		self.apply(weights_init_)

	def forward(self, state, action):
		xu = torch.cat([state, action], 1)
        
		x1 = F.relu(self.linear1(xu))
		x1 = F.relu(self.linear2(x1))
		x1 = self.linear3(x1)

		x2 = F.relu(self.linear4(xu))
		x2 = F.relu(self.linear5(x2))
		x2 = self.linear6(x2)
		return x1, x2

	def loss(self, memory, batch_size, optimistic=False, state=None, action=None, eta=None):
		states, actions, qvalues=memory.sample(batch_size)
		states = torch.FloatTensor(states).to(device)
		actions = torch.FloatTensor(actions).to(device)
		qvalues = torch.FloatTensor(qvalues).to(device)
		predict_qvalues1, predict_qvalues2 = self.forward(states, actions)
		mse_loss = F.mse_loss(predict_qvalues1.reshape(-1), qvalues) + F.mse_loss(predict_qvalues2.reshape(-1), qvalues)
		if optimistic==False:
			return mse_loss
		if eta is None:
			eta=1.0

		reward_loss1, reward_loss2=self.forward(state, action)
		return -eta * torch.min(reward_loss1, reward_loss2) + mse_loss





