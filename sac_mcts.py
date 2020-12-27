# -*- coding: utf-8 -*-
import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory, TrajReplayMemory
import time
import copy
from critic import QNetwork, DoubleQNetwork
from policy import GaussianPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import soft_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Humanoid-v2",
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
parser.add_argument('--alg_name', default="ROFU_SAC")
parser.add_argument('--max_interact_steps', default=1000)
parser.add_argument('--truncate_length', default=100)
parser.add_argument('--double_Q', default=True)
parser.add_argument('--explore_rate', default = 1.)
parser.add_argument("--result_folder", default='results')
parser.add_argument("--eta_init", type=float, default=3.)
parser.add_argument("--train_freq", type=int, default=10)
parser.add_argument("--non_optimistic_steps", type=int, default=1)
parser.add_argument('--rofu_steps', type=int, default=3)
args = parser.parse_args()
filename = f"./{args.result_folder}/{args.env_name}_seed_{args.seed}"

print(filename)

writer=SummaryWriter(comment=f"{args.env_name}_{args.alg_name}")

epi=0
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]

class ActorCriticMCTS:
	def __init__(self, policy, critic, env, memory, args):
		self.policy=policy
		self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
		self.critic=critic
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
		self.critic_target = copy.deepcopy(self.critic)
		self.alpha = args.alpha
		self.gamma = args.gamma
		self.memory=memory
		self.env=env
		self.critic_upd_steps = 0
		self.explore_rate = args.explore_rate
		self.samples = 0

		self.samples_log = []
		self.evals_log = []


	def clear(self):
		policy = GaussianPolicy(state_dim, action_dim).to(device=device)
		critic = QNetwork(state_dim, action_dim, args.hidden_size).to(device=device)

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
			return actions[0], 0.

		optimism_r = []
		critic_evals = []
		optimistic_evals = []
		for action in actions:
			rofurew, critic_eval, optimistic_eval = self.RofuReward(tensor_state, action, training_steps=args.rofu_steps)
			optimism_r.append(rofurew)
			critic_evals.append(critic_eval)
			optimistic_evals.append(optimistic_eval)

		action_id = np.argmin(optimism_r)
		return actions[action_id], optimistic_evals[action_id] #np.min(optimism_r)

	def compute_eta(self):
		return args.eta_init / np.log(1. + len(self.memory))#1./ np.sqrt(1. + len(self.memory))

	def RofuReward(self, state, action, training_steps=10):
		critic_state_dict = copy.deepcopy(self.critic.state_dict())
		optimizer_state_dict = copy.deepcopy(self.critic_optimizer.state_dict())
		eta = self.compute_eta()
		for s in range(training_steps):
			loss = self.rofu_loss(self.critic, eta, state, action)
			self.critic_optimizer.zero_grad()
			loss.backward()
			self.critic_optimizer.step()
		loss = self.rofu_loss(self.critic, eta, state, action).cpu().data.numpy()
		r1, r2 = self.critic(state.detach(), action.detach())
		optimistic_est = torch.min(r1, r2).cpu().data.numpy()
		self.critic.load_state_dict(critic_state_dict)
		self.critic_optimizer.load_state_dict(optimizer_state_dict)
		cr1, cr2 = self.critic(state.detach(), action.detach())
		cri_est = torch.min(cr1, cr2).cpu().data.numpy()
		return loss, cri_est, optimistic_est

	# 	optimism_critic=copy.deepcopy(self.critic)
	# 	optimizer=torch.optim.Adam(optimism_critic.parameters(), lr=1e-3)
	# 	eta=self.compute_eta()
	# 	for s in range(training_steps):
	# 		#loss=optimism_critic.loss(self.memory, args.batch_size, optimistic=True,\
	# 		#state=state.detach(), action=action.detach(), eta=eta)
	# 		loss = self.rofu_loss(optimism_critic, eta, state, action)
	# 		optimizer.zero_grad()
	# 		loss.backward()
	# 		optimizer.step()
	# 	rofu_loss = self.rofu_loss(optimism_critic, eta, state, action)
	# 	return rofu_loss, self.critic(state, action),  optimism_critic(state, action)

	def rofu_loss(self, optimism_critic, eta, state, action):
		td_loss = self.td_critic_loss(optimism_critic)
		r1, r2 = optimism_critic(state.detach(), action.detach())
		reward_loss= -torch.min(r1, r2)
		loss = reward_loss * eta + td_loss
		return loss

	def Interaction(self, state, steps, max_steps, done, rand_act=False, optimistic=False):
		if steps==max_steps or done:
			return [state], [], [], []
		if rand_act:
			action, estimated = self.env.action_space.sample(), [0.]
		elif optimistic:
			if self.samples < args.non_optimistic_steps:
				action, ucb = self.SelectAction(state, num_candidates=1)
			else:
				action, ucb = self.SelectAction(state, num_candidates=5)
			action=action.cpu().data.numpy()[0]
			estimated=[ucb]
		else:
			raise NotImplemented
		next_state, reward, done, _ = self.env.step(action)
		path_state = [state]
		path_action = [action]
		path_reward = [reward]

		self.samples += 1
		if self.samples % 1000 == 0:
			curres = []
			_env = gym.make(args.env_name)
			for __ in range(5):
				state = _env.reset()
				evaluations = self.eval(state, _env)
				curres.append(evaluations)
			self.evals_log.append(curres)
			self.samples_log.append(self.samples)
			np.savez(filename, samples=self.samples_log, evals=self.evals_log)
			writer.add_scalar("evaluations", np.mean(curres), self.samples)

		_s, _a, _r, est = self.Interaction(next_state, steps+1, max_steps, done, rand_act=rand_act, optimistic=optimistic)
		tensor_state=torch.FloatTensor(state).reshape(1, -1).to(device)
		tensor_action = torch.FloatTensor(action).reshape(1, -1).to(device)
		if optimistic:
			disc_r = reward
			_g = 1.
			for __ in _r:
				_g *= self.gamma
				disc_r += _g * __
			cr1, cr2 = self.critic(tensor_state, tensor_action)
			cr = torch.min(cr1, cr2).cpu().data.numpy()
			print("selecting action", steps, disc_r, cr, estimated[0])
		return path_state + _s, path_action +_a, path_reward + _r, estimated+est

	def eval(self, state, _env, steps=0, max_steps=1000, done=False):
		if steps == max_steps or done:
			return 0.
		tensor_state = torch.FloatTensor(state).reshape(1, -1).to(device)
		_action = self.policy.sample(tensor_state)[2]
		action = _action.cpu().data.numpy()[0]
		next_state, reward, done, _ = _env.step(action)
		#print("eval steps", steps, reward)
		q = self.eval(next_state, _env, steps + 1, max_steps, done)
		return q + reward


	def td_critic_loss(self, critic, batch_size=256):
		states, actions, rewards, next_states, masks = self.memory.sample(batch_size)
		states = torch.FloatTensor(states).to(device)
		actions = torch.FloatTensor(actions).to(device)
		rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
		next_states = torch.FloatTensor(next_states).to(device)
		masks = torch.FloatTensor(masks).reshape(-1, 1).to(device)

		with torch.no_grad():
			n_action, n_a_log_prob, _ = self.policy.sample(next_states)
			n_qf1, n_qf2 = self.critic_target(next_states, n_action)
			n_qf = torch.min(n_qf1,n_qf2) - self.alpha * n_a_log_prob
			n_qvalue = rewards + self.gamma * masks * n_qf

		qf1, qf2 = critic(states, actions)
		qf1_loss = F.mse_loss(qf1, n_qvalue)
		qf2_loss = F.mse_loss(qf2, n_qvalue)
		qf_loss = qf1_loss + qf2_loss

		return qf_loss

	def train_critic(self):
		loss = self.td_critic_loss(self.critic)
		self.critic_optimizer.zero_grad()
		self.critic_upd_steps += 1
		writer.add_scalar("critic loss", loss, self.critic_upd_steps)
		loss.backward()
		self.critic_optimizer.step()
		soft_update(self.critic_target, self.critic, args.tau)
		return loss

	def train_policy(self):
		states, actions, rewards, next_states, masks=self.memory.sample(args.batch_size)
		states = torch.FloatTensor(states).to(device)
		actions = torch.FloatTensor(actions).to(device)
		rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
		next_states = torch.FloatTensor(next_states).to(device)
		masks = torch.FloatTensor(masks).reshape(-1, 1).to(device)
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

def parse_path(path_state, path_action, path_reward, replay_memory, max_steps=args.max_interact_steps, truncate=False, truncate_length=50):
	length = len(path_state)
	for i in range(length - 1):
		mask = 1.
		if i == length - 2 and length <= max_steps:
			mask=0.
		replay_memory.push(path_state[i], path_action[i], path_reward[i], path_state[i + 1], mask)

def implicit_confidence_bound(mcts, n=20):		
	states, actions, rewards, next_states, masks=self.memory.sample(n)
	states = torch.FloatTensor(states).to(device)
	actions = torch.FloatTensor(actions).to(device)
	rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
	next_states = torch.FloatTensor(next_states).to(device)
	masks = torch.FloatTensor(masks).to(device)
	res = 0.
	for i in range(n):
		tensor_state = states[i].reshape(1, -1).to(device)
		tensor_action = actions[i].reshape(1, -1).to(device)
		rofur, cr, ocr = mcts.RofuReward(tensor_state, tensor_action)
		res += ocr- cr
	return res / n

def policy_var(policy, mcts, n=200):
	states, actions, rewards, next_states, masks=mcts.memory.sample(n)
	states = torch.FloatTensor(states).to(device)
	actions, _, mean = policy.sample(states)
	return (actions-mean).norm(2) / n

policy = GaussianPolicy(state_dim, action_dim).to(device=device)
if args.double_Q:
	critic = DoubleQNetwork(state_dim, action_dim, args.hidden_size).to(device=device)
else:
	critic = QNetwork(state_dim, action_dim, args.hidden_size).to(device=device)

memory = TrajReplayMemory(args.replay_size, args.seed, state_dim, action_dim)

alg = ActorCriticMCTS(policy, critic, env, memory, args)
__=0
while len(memory) < args.start_steps:
	state=env.reset()
	ps, pa, pr, _=alg.Interaction(state, done=False, steps=0, max_steps=10, rand_act=True)
	print(len(ps), len(pa), len(pr))
	parse_path(ps, pa, pr, memory)
	writer.add_scalar("rand reward", np.sum(pr), __)
	__+=1
	break

critic_train_steps=0

alg.samples = len(memory)
while alg.samples < args.num_steps:
	epi += 1
	#if epi % 1 == 0:
	#	alg.clear()
	if epi % 1 == 0: 
		pass

	state = env.reset()
	print("episode", epi)
	ps, pa, pr,_ = alg.Interaction(state, done=False, steps=0, max_steps=args.max_interact_steps, optimistic=True)
	#print(ps, pa, pr)
	parse_path(ps, pa, pr, memory, truncate=True, truncate_length=args.max_interact_steps)
	#samples += len(ps)
	writer.add_scalar("epireward", np.sum(pr), epi)
	interaction_steps=len(ps)
	train_freq=args.train_freq if alg.samples > args.non_optimistic_steps else 1
	for _ in range(train_freq * len(ps)):
		alg.train_critic()
	p_ent_loss, p_rew_loss =0, 0
	for _ in range(train_freq * len(ps)):
		p_ent_loss, p_rew_loss=alg.train_policy()
	p_var = policy_var(alg.policy, alg)
	writer.add_scalar("policy_var", p_var, alg.samples)
	writer.add_scalar("policy_ent_loss", p_ent_loss, alg.samples)
	writer.add_scalar("policy_rew_loss", p_rew_loss, alg.samples)
