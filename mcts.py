# -*- coding: utf-8 -*-
import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import time
import copy
from critic import QNetwork, DoubleQNetwork
from policy import GaussianPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Hopper-v2",
					help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
					help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
					help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
					help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
					help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=.5, metavar='G',
					help='Temperature parameter α determines the relative importance of the entropy\
							term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
					help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=6, metavar='N',
					help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
					help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
					help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=400, metavar='N',
					help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=20, metavar='N',
					help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
					help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
					help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
					help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
					help='run on CUDA (default: False)')
parser.add_argument('--model_upd_freq', default=100)
parser.add_argument('--rollout_len', default=1)
parser.add_argument('--rollout_per_step', default=400)
parser.add_argument('--policy_pretrain_steps', default=500000)
parser.add_argument('--model_pretrain_steps', default=10000)
parser.add_argument('--model_resize', default=False)
parser.add_argument('--alg_name', default="MCTS")
parser.add_argument('--max_interact_steps', default=1000)
parser.add_argument('--truncate_length', default=100)
parser.add_argument('--double_Q', default=True)
args = parser.parse_args()


writer=SummaryWriter(comment=args.alg_name)



env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]

class ActorCriticMCTS:
	def __init__(self, policy, critic, env, memory):
		self.policy=policy
		self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
		self.critic=critic
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.memory=memory
		self.env=env

	def clear(self):
		#policy = GaussianPolicy(state_dim, action_dim).to(device=device)
		#critic = QNetwork(state_dim, action_dim, args.hidden_size).to(device=device)

		self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		for _ in range(3000):
			self.train_critic()
		for _ in range(1000):
			self.train_policy()


	def SelectAction(self, state, num_candidates):
		tensor_state = torch.FloatTensor(state).reshape(1, -1).to(device)
		actions=[self.policy.sample(tensor_state)[0] for _ in range(num_candidates)]
		if num_candidates == 1:
			return actions[0], torch.zeros(1).to(device)

		optimism_r = []
		critic_evals = []
		optimistic_evals = []
		for action in actions:
			rofurew, critic_eval, optimistic_eval = self.RofuReward(tensor_state, action)
			optimism_r.append(rofurew)
			critic_evals.append(critic_eval)
			optimistic_evals.append(optimistic_eval)

		action_id = np.argmin(optimism_r)
		return actions[action_id], np.min(optimism_r)

	def compute_eta(self):
		return 1./ np.sqrt(1. + len(self.memory))

	def RofuReward(self, state, action, training_steps=10):
		optimism_critic=copy.deepcopy(self.critic)
		optimizer=torch.optim.Adam(optimism_critic.parameters(), lr=3e-4)
		eta=self.compute_eta()
		for s in range(training_steps):
			loss=optimism_critic.loss(self.memory, args.batch_size, optimistic=True,\
			 	state=state.detach(), action=action.detach(), eta=eta)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		rofurew=optimism_critic.loss(self.memory, args.batch_size, optimistic=True, state=state, action=action)
		return rofurew, self.critic(state, action), optimism_critic(state, action)


	def Interaction(self, state, steps, max_steps, done, rand_act=False, optimistic=False):
		if steps==max_steps or done:
			return [], [], [], []
		if rand_act:
			action, estimated = self.env.action_space.sample(), [0.]
		elif optimistic:
			action, ucb = self.SelectAction(state, num_candidates=10)
			action=action.cpu().data.numpy()[0]
			estimated=[ucb]
		else:
			raise NotImplemented
		next_state, reward, done, _ = self.env.step(action)
		path_state = [state]
		path_action = [action]
		path_reward = [reward]

		_s, _a, _r, est = self.Interaction(next_state, steps+1, max_steps, done, rand_act=rand_act, optimistic=optimistic)
		tensor_state=torch.FloatTensor(state).reshape(1, -1).to(device)
		tensor_action = torch.FloatTensor(action).reshape(1, -1).to(device)
		if optimistic:
			if args.double_Q:
				rew1, rew2 = self.critic(tensor_state, tensor_action)
				print("steps", steps, reward + np.sum(_r[:args.truncate_length]), torch.min(rew1, rew2).cpu().data.numpy(), -estimated[0].cpu().data.numpy())				
			else:
				print("steps", steps, -estimated[0].cpu().data.numpy(), reward + np.sum(_r[:args.truncate_length]), self.critic(tensor_state, tensor_action).cpu().data.numpy())
		return path_state + _s, path_action +_a, path_reward + _r, estimated+est

	def eval(self, state, steps=0, max_steps=1000, done=False, num_candidates=1):
		if steps == max_steps or done:
			return 0.
		tensor_state = torch.FloatTensor(state).reshape(1, -1).to(device)
		action = None 
		estedq=-10000.
		for a in range(num_candidates):
			_action = self.policy.sample(tensor_state)[2]
			if num_candidates == 1:
				action = _action.cpu().data.numpy()[0]
				break
			q_sa = self.critic(tensor_state, _action)
			if q_sa > estedq:
				action = _action.cpu().data.numpy()[0]
				estedq = q_sa
		next_state, reward, done, _ = self.env.step(action)
		print("eval steps", steps, reward)
		q = self.eval(next_state, steps + 1, max_steps, done)
		return q + reward


	def train_critic(self):
		loss = self.critic.loss(self.memory, args.batch_size)
		self.critic_optimizer.zero_grad()
		loss.backward()
		self.critic_optimizer.step()
		return loss

	def train_policy(self):
		states, actions, qvalues=self.memory.sample(args.batch_size)
		states = torch.FloatTensor(states).to(device)
		actions = torch.FloatTensor(actions).to(device)
		qvalues = torch.FloatTensor(qvalues).to(device)
		p_actions, log_prob, _=self.policy.sample(states)

		ent_loss = args.alpha * log_prob.mean()
		if args.double_Q:
			rew1, rew2 = self.critic(states, p_actions)
			rew_loss = -torch.min(rew1, rew2).mean()
		else:
			rew_loss = -self.critic(states, p_actions).mean()
		loss = ent_loss + rew_loss
		
		self.policy_optimizer.zero_grad()
		loss.backward()
		self.policy_optimizer.step()
		return ent_loss, rew_loss

def parse_path(path_state, path_action, path_reward, replay_memory, truncate=False, truncate_length=50):
	length=len(path_state)
	q_values=np.zeros(length)
	q_values[-1]=path_reward[-1]

	for i in range(2, length+1):
		q_values[-i]=path_reward[-i] + q_values[-(i-1)]
	if truncate:
		tail = length - truncate_length + 1
		if length < args.max_interact_steps:
			tail = length
		for i in range(0, tail):
			end = i + truncate_length
			tail_q = 0. if end >= length else q_values[end]
			replay_memory.push(path_state[i], path_action[i], q_values[i] - tail_q)
			print("push values", i, q_values[i] - tail_q)

	else:
		for i in range(length):
			replay_memory.push(path_state[i], path_action[i], q_values[i])

def implicit_confidence_bound(mcts, n=20):
	states, actions, qvalues=mcts.memory.sample(n)
	res = 0.
	for i in range(n):
		tensor_state = torch.FloatTensor(states[i]).reshape(1, -1).to(device)
		tensor_action = torch.FloatTensor(actions[i]).reshape(1, -1).to(device)
		rofur, cr, ocr = mcts.RofuReward(tensor_state, tensor_action)
		res += ocr- cr
	return res / n

def policy_var(policy, mcts, n=200):
	states, actions, qvalues=mcts.memory.sample(n)
	states = torch.FloatTensor(states).to(device)
	actions, _, mean = policy.sample(states)
	return (actions-mean).norm(2) / n

policy = GaussianPolicy(state_dim, action_dim).to(device=device)
if args.double_Q:
	critic = DoubleQNetwork(state_dim, action_dim, args.hidden_size).to(device=device)
else:
	critic = QNetwork(state_dim, action_dim, args.hidden_size).to(device=device)

memory = ReplayMemory(20000, args.seed, state_dim, action_dim)

alg = ActorCriticMCTS(policy, critic, env, memory)
__=0
while len(memory) < 300:
	state=env.reset()
	ps, pa, pr, _=alg.Interaction(state, done=False, steps=0, max_steps=args.max_interact_steps, rand_act=True)
	parse_path(ps, pa, pr, memory)
	writer.add_scalar("rand reward", np.sum(pr), __)
	__+=1

critic_train_steps=0
for i in range(1000):
	loss=alg.train_critic()
	writer.add_scalar("critic_loss", loss, critic_train_steps)
	critic_train_steps += 1
policy_train_steps=0
for i in range(1000):
	ent_loss, rew_loss=alg.train_policy()
	writer.add_scalar("policy_ent_loss", ent_loss / args.alpha, policy_train_steps)
	writer.add_scalar("policy_rew_loss", rew_loss, policy_train_steps)
	policy_train_steps += 1

samples = 300
for epi in range(10000):
	if epi % 1 == 0:
		alg.clear()
	if args.double_Q==False:
		cnfb = implicit_confidence_bound(alg)
		writer.add_scalar("implicit_confidence_bound", cnfb, samples)
	state = env.reset()
	evaluations = alg.eval(state)
	writer.add_scalar("evaluations", evaluations, samples)
	state = env.reset()
	print("episode", epi)
	ps, pa, pr, ucb = alg.Interaction(state, done=False, steps=0, max_steps=args.max_interact_steps, optimistic=True)
	parse_path(ps, pa, pr, memory, truncate=True, truncate_length=args.truncate_length)
	samples += len(ps)
	writer.add_scalar("epireward", np.sum(pr), epi)
	interaction_steps=len(ps)
	max_training_steps = 100 * interaction_steps
	min_training_steps = 20 * interaction_steps
	cur_critic_train_steps = 0
	c_loss = 100000.
	while cur_critic_train_steps < max_training_steps:
		c_loss=alg.train_critic()
		cur_critic_train_steps += 1
		critic_train_steps += 1
		writer.add_scalar("critic_loss", c_loss, critic_train_steps)

	for _ in range(interaction_steps):
		
		p_e_loss, p_r_loss=alg.train_policy()
		policy_train_steps += 1
		writer.add_scalar("policy_ent_loss", p_e_loss / args.alpha, policy_train_steps)
		writer.add_scalar("policy_rew_loss", p_r_loss, policy_train_steps)
		writer.add_scalar("policy_total_loss", args.alpha * p_e_loss + p_r_loss, policy_train_steps)
		pvar = policy_var(policy, alg)
		writer.add_scalar("policy_variance", pvar, policy_train_steps)