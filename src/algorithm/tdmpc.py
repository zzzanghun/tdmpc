import os
import glob
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import algorithm.helper as h
from collections import deque, OrderedDict

PROJECT_HOME = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'results')

class TOLD(nn.Module):
	"""Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.elu = nn.ELU()
		self.sigmoid = nn.Sigmoid()
		self._encoder = h.enc(cfg)
		if cfg.CURIOSITY_ENCODER:
			self._curiosity_encoder = h.enc(cfg)
		self._dynamics = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
		self._reward = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
		if cfg.PI_PARAMETERIZED:
			self._pi = nn.Sequential(
				nn.Linear(cfg.latent_dim, cfg.mlp_dim),
				nn.ELU(),
				nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
				nn.ELU()
			)
			self._original_parameterized_net = nn.Linear(512, cfg.action_dim)
			self._scale_parameterized_net = nn.Sequential(nn.Linear(512, cfg.action_dim),
													nn.Sigmoid())
			self._bias_parameterized_net = nn.Linear(512, cfg.action_dim)
		elif cfg.PI_EACH_PARAMETERIZED:
			self._pi = nn.Sequential(
				nn.Linear(cfg.latent_dim, cfg.mlp_dim),
				nn.ELU(),
				nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
				nn.ELU()
			)
			policy_parameterized_net_dict = OrderedDict()
			for i in range(cfg.action_dim):
				policy_parameterized_net_dict["policy_parameterized_{0}".format(i)] = nn.Linear(cfg.mlp_dim, 1)
			self._pi_each_parameterized_net = nn.Sequential(policy_parameterized_net_dict)
		else:
			self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
		self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
		self.apply(h.orthogonal_init)
		for m in [self._reward, self._Q1, self._Q2]:
			m[-1].weight.data.fill_(0)
			m[-1].bias.data.fill_(0)

	def track_q_grad(self, enable=True):
		"""Utility function. Enables/disables gradient tracking of Q-networks."""
		for m in [self._Q1, self._Q2]:
			h.set_requires_grad(m, enable)

	def h(self, obs, int_reward=False):
		"""Encodes an observation into its latent representation (h)."""
		if self.cfg.CURIOSITY_ENCODER and int_reward:
			return self._curiosity_encoder(obs)
		else:
			return self._encoder(obs)

	def next(self, z, a):
		"""Predicts next latent state (d) and single-step reward (R)."""
		x = torch.cat([z, a], dim=-1)
		return self._dynamics(x), self._reward(x)

	def pi(self, z, std=0):
		"""Samples an action from the learned policy (pi)."""
		if self.cfg.PI_PARAMETERIZED:
			z = self._pi(z)
			a = self._original_parameterized_net(z)
			k = self._scale_parameterized_net(z)
			a0 = self._bias_parameterized_net(z)
			mu = torch.tanh(k * a - k * a0)
		elif self.cfg.PI_EACH_PARAMETERIZED:
			mu = torch.zeros((z.shape[0], self.cfg.action_dim), dtype=torch.float32, device=z.device)
			z = self._pi(z)
			for i in range(len(self._pi_each_parameterized_net)):
				action_parameterized = self._pi_each_parameterized_net[i](z)
				mu[:, i] = action_parameterized[:, 0]
			mu = torch.tanh(mu)
		else:
			mu = torch.tanh(self._pi(z))
		if std > 0:
			std = torch.ones_like(mu) * std
			return h.TruncatedNormal(mu, std).sample(clip=0.3)
		return mu

	def Q(self, z, a):
		"""Predict state-action value (Q)."""
		x = torch.cat([z, a], dim=-1)
		return self._Q1(x), self._Q2(x)


class TDMPC():
	"""Implementation of TD-MPC learning + inference."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		self.std = h.linear_schedule(cfg.std_schedule, 0)
		self.epsilon = h.linear_schedule(cfg.std_schedule, 0)
		self.model = TOLD(cfg).to(self.device)
		self.model_target = deepcopy(self.model)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
		if cfg.PI_PARAMETERIZED:
			self.pi_optim = torch.optim.Adam([
				{'params': self.model._pi.parameters()},
				{'params': self.model._original_parameterized_net.parameters()},
				{'params': self.model._scale_parameterized_net.parameters()},
				{'params': self.model._bias_parameterized_net.parameters()}
				], lr=self.cfg.lr)
		elif cfg.PI_EACH_PARAMETERIZED:
			self.pi_optim = torch.optim.Adam([
				{'params': self.model._pi.parameters()},
				{'params': self.model._pi_each_parameterized_net.parameters()}
				], lr=self.cfg.lr)
		else:
			self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
		if cfg.CURIOSITY_ENCODER:
			self.curiosity_encoder_optim = torch.optim.Adam(self.model._curiosity_encoder.parameters(), lr=self.cfg.lr)
		self.aug = h.RandomShiftsAug(cfg)
		self.action_type = deque(maxlen=cfg.episode_length)
		self.valuable_action = deque(maxlen=cfg.episode_length)
		self.model.eval()
		self.model_target.eval()
		self.prev_obs = None
		self.mse_loss = nn.MSELoss()

		if cfg.JUDGE_Q:
			self.model_for_judge_q = deepcopy(self.model)
			self.model_save_dir = os.path.join(PROJECT_HOME, "models", self.cfg.domain + "-" + self.cfg.task)
			self.load_judge_q()
		else:
			self.judge_q = None

		if self.cfg.CHOICE_ACTION_POLICY_AND_PLAN_BY_Q:
			self.choice_action_start_step = int(int(cfg.train_steps) / 5)
		elif self.cfg.CHOICE_ACTION_POLICY_AND_PLAN_BY_EPSILON:
			self.choice_action_start_step = int(int(cfg.train_steps) / 10)

	def state_dict(self):
		"""Retrieve state dict of TOLD model, including slow-moving target network."""
		return {'model': self.model.state_dict(),
				'model_target': self.model_target.state_dict()}

	def save(self, fp):
		"""Save state dict of TOLD model to filepath."""
		torch.save(self.state_dict(), fp)
	
	def load(self, fp):
		"""Load a saved state dict from filepath into current agent."""
		d = torch.load(fp)
		self.model.load_state_dict(d['model'])
		self.model_target.load_state_dict(d['model_target'])

	@torch.no_grad()
	def estimate_value(self, z, actions, horizon):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(horizon):
			z, reward = self.model.next(z, actions[t])
			G += discount * reward
			discount *= self.cfg.discount
		G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
		return G

	@torch.no_grad()
	def plan(self, obs, eval_mode=False, step=None, t0=True):
		"""
		Plan next action using TD-MPC inference.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		"""
		# Seed steps
		self.prev_obs = obs
		if step < self.cfg.seed_steps and not eval_mode:
			return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

		# Sample policy trajectories
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
		num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
		if num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
			z = self.model.h(obs).repeat(num_pi_trajs, 1)
			for t in range(horizon):
				pi_actions[t] = self.model.pi(z, self.cfg.min_std)
				z, _ = self.model.next(z, pi_actions[t])

		# Initialize state and parameters
		z = self.model.h(obs).repeat(self.cfg.num_samples+num_pi_trajs, 1)
		mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
		std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device)
		if not t0 and hasattr(self, '_prev_mean'):
			mean[:-1] = self._prev_mean[1:]

		# Iterate CEM
		for i in range(self.cfg.iterations):
			actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device), -1, 1)
			if num_pi_trajs > 0:
				actions = torch.cat([actions, pi_actions], dim=1)

			# Compute elite actions
			value = self.estimate_value(z, actions, horizon)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			_std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
			_std = _std.clamp_(self.std, 2)
			mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

		# Outputs
		score = score.squeeze(1).cpu().numpy()
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		self._prev_mean = mean
		mean, std = actions[0], _std[0]

		a = mean
		if not eval_mode:
			a += std * torch.randn(self.cfg.action_dim, device=std.device)
			pi_action = self.model.pi(torch.unsqueeze(z[0], 0), self.std)
		else:
			pi_action = self.model.pi(torch.unsqueeze(z[0], 0))

		if self.cfg.JUDGE_Q:
			pi_action_judge_q = torch.min(*self.model_for_judge_q.Q(torch.unsqueeze(z[0], 0), pi_action))
			plan_action_judge_q = torch.min(*self.model_for_judge_q.Q(torch.unsqueeze(z[0], 0), torch.unsqueeze(a, 0)))
			if pi_action_judge_q > plan_action_judge_q:
				self.valuable_action.append(1)
			else:
				self.valuable_action.append(0)

		if self.cfg.COMPARISON_TEST:
			if eval_mode:
				self.action_type.append(1)
				return pi_action[0]

		if self.cfg.CHOICE_ACTION_POLICY_AND_PLAN_BY_Q:
			if step >= self.choice_action_start_step:
				pi_action_q = torch.min(*self.model.Q(torch.unsqueeze(z[0], 0), pi_action))
				plan_action_q = torch.min(*self.model.Q(torch.unsqueeze(z[0], 0), torch.unsqueeze(a, 0)))
				if pi_action_q > plan_action_q:
					self.action_type.append(1)
					return pi_action[0]

		elif self.cfg.CHOICE_ACTION_POLICY_AND_PLAN_BY_EPSILON:
			if t0:
				self.epsilon = h.linear_schedule(self.cfg.epsilon_schedule, step*self.cfg.action_repeat)
			coin = np.random.random()  # 0 ~ 1
			if coin > self.epsilon:
				if step >= self.choice_action_start_step:
					pi_action_q = torch.min(*self.model.Q(torch.unsqueeze(z[0], 0), pi_action))
					plan_action_q = torch.min(*self.model.Q(torch.unsqueeze(z[0], 0), torch.unsqueeze(a, 0)))
					if pi_action_q > plan_action_q:
						self.action_type.append(1)
						return pi_action[0]

		elif self.cfg.CHOICE_ACTION_POLICY_AND_PLAN_BY_FRONT:
			if t0:
				self.epsilon = h.linear_schedule(self.cfg.epsilon_schedule, step*self.cfg.action_repeat)
			coin = np.random.random()  # 0 ~ 1
			if coin <= self.epsilon:
				self.action_type.append(0)
				return a
			else:
				self.action_type.append(1)
				return pi_action[0]

		elif self.cfg.CHOICE_ACTION_POLICY_AND_PLAN_BY_REVERSE:
			if t0:
				self.epsilon = h.linear_schedule(self.cfg.epsilon_schedule, step*self.cfg.action_repeat)
			coin = np.random.random()  # 0 ~ 1
			if coin <= self.epsilon:
				self.action_type.append(1)
				return pi_action[0]

		self.action_type.append(0)
		return a

	def update_pi(self, zs):
		"""Update policy using a sequence of latent states."""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)

		# Loss is a weighted sum of Q-values
		pi_loss = 0
		for t,z in enumerate(zs):
			a = self.model.pi(z, self.cfg.min_std)
			Q = torch.min(*self.model.Q(z, a))
			pi_loss += -Q.mean() * (self.cfg.rho ** t)

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		return pi_loss.item()

	def update_curiosity_encoder(self, obs, next_obses, action):
		self.curiosity_encoder_optim.zero_grad(set_to_none=True)
		z = self.model.h(self.aug(obs), int_reward=True)

		consistency_loss = 0

		for t in range(self.cfg.horizon):
			# Predictions
			z, _ = self.model.next(z, action[t])
			with torch.no_grad():
				next_obs = self.aug(next_obses[t])
				next_z = self.model_target.h(next_obs, int_reward=True)

			# Losses
			rho = (self.cfg.rho ** t)
			consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
		consistency_loss = self.cfg.CURIOSITY_ENCODER_COEF * consistency_loss.clamp(max=1e4)
		consistency_loss = consistency_loss.mean()
		consistency_loss.register_hook(lambda grad: grad * (1 / self.cfg.horizon))
		consistency_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._curiosity_encoder.parameters(), self.cfg.grad_clip_norm,
												   error_if_nonfinite=False)
		self.curiosity_encoder_optim.step()
		return consistency_loss.item()

	@torch.no_grad()
	def _td_target(self, next_obs, reward):
		"""Compute the TD-target from a reward and the observation at the following time step."""
		next_z = self.model.h(next_obs)
		td_target = reward + self.cfg.discount * \
			torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))
		return td_target

	def update(self, replay_buffer, step):
		"""Main update function. Corresponds to one iteration of the TOLD model learning."""
		obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.std = h.linear_schedule(self.cfg.std_schedule, step)
		self.model.train()

		# Representation
		z = self.model.h(self.aug(obs))
		zs = [z.detach()]

		consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
		for t in range(self.cfg.horizon):

			# Predictions
			Q1, Q2 = self.model.Q(z, action[t])
			z, reward_pred = self.model.next(z, action[t])
			with torch.no_grad():
				next_obs = self.aug(next_obses[t])
				next_z = self.model_target.h(next_obs)
				td_target = self._td_target(next_obs, reward[t])
			zs.append(z.detach())

			# Losses
			rho = (self.cfg.rho ** t)
			consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
			reward_loss += rho * h.mse(reward_pred, reward[t])
			value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
			priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))

		# Optimize model
		total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
					 self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
					 self.cfg.value_coef * value_loss.clamp(max=1e4)
		weighted_loss = (total_loss * weights).mean()
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		weighted_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()
		replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

		# Update policy + target network
		pi_loss = self.update_pi(zs)
		if self.cfg.CURIOSITY_ENCODER and step % self.cfg.episode_length == 0:
			curiosity_encoder_loss = self.update_curiosity_encoder(obs, next_obses, action)
		if step % self.cfg.update_freq == 0:
			h.ema(self.model, self.model_target, self.cfg.tau)

		self.model.eval()

		if len(self.action_type) > 0:
			mean_action_type = float(sum(self.action_type)/len(self.action_type))
		else:
			mean_action_type = 0.0

		if len(self.valuable_action) > 0:
			mean_valuable_action = float(sum(self.valuable_action) / len(self.valuable_action))
		else:
			mean_valuable_action = 0.0

		if self.cfg.CURIOSITY_ENCODER and step % self.cfg.episode_length == 0:
			return {'consistency_loss': float(consistency_loss.mean().item()),
					'reward_loss': float(reward_loss.mean().item()),
					'value_loss': float(value_loss.mean().item()),
					'pi_loss': pi_loss,
					'total_loss': float(total_loss.mean().item()),
					'weighted_loss': float(weighted_loss.mean().item()),
					'grad_norm': float(grad_norm),
					'action_type': mean_action_type,
					'valuable_action': mean_valuable_action,
					'curiosity_encoder_loss': curiosity_encoder_loss}
		else:
			return {'consistency_loss': float(consistency_loss.mean().item()),
					'reward_loss': float(reward_loss.mean().item()),
					'value_loss': float(value_loss.mean().item()),
					'pi_loss': pi_loss,
					'total_loss': float(total_loss.mean().item()),
					'weighted_loss': float(weighted_loss.mean().item()),
					'grad_norm': float(grad_norm),
					'action_type': mean_action_type,
					'valuable_action': mean_valuable_action}

	def calc_int_reward(self, obs, action):
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		self.prev_obs = torch.tensor(self.prev_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

		if self.cfg.CURIOSITY_ENCODER:
			real_next_inputs_feature = self.model.h(obs, int_reward=True)

			real_current_inputs_feature = self.model.h(self.prev_obs, int_reward=True)
			pred_next_inputs_feature, _ = self.model.next(real_current_inputs_feature, action.unsqueeze(0))
			prediction_error = torch.mean(h.mse(pred_next_inputs_feature, real_next_inputs_feature), dim=1,
										  keepdim=True).mean()
		elif self.cfg.Q_CURIOSITY:
			real_current_inputs_feature = self.model.h(self.prev_obs)
			Q_value, _ = self.model.Q(real_current_inputs_feature, action.unsqueeze(0))
			target_Q_value, _ = self.model_target.Q(real_current_inputs_feature, action.unsqueeze(0))
			prediction_error = torch.mean(h.mse(Q_value, target_Q_value), dim=1, keepdim=True).mean()
		else:
			real_next_inputs_feature = self.model.h(obs)

			real_current_inputs_feature = self.model.h(self.prev_obs)
			pred_next_inputs_feature, _ = self.model.next(real_current_inputs_feature, action.unsqueeze(0))
			prediction_error = torch.mean(h.mse(pred_next_inputs_feature, real_next_inputs_feature), dim=1,
										  keepdim=True).mean()

		int_rewards = self.cfg.BETA * 0.5 * prediction_error

		return int_rewards.detach()

	def load_judge_q(self):
		model_file_dict = {}
		model_file_list = glob.glob(os.path.join(self.model_save_dir, "*.pth"))
		idx = 1
		for model_file_name in model_file_list:
			print("{0}.".format(idx))
			print("{0}".format(model_file_name))
			model_file_dict[idx] = model_file_name
			idx += 1
		try:
			chosen_number = int(
				input("Choose ONE NUMBER from the above options and press enter (two or more times) to continue..."))
		except ValueError as e:
			chosen_number = 0

		if chosen_number == 0:
			print("### START WITH *RANDOM* DEEP LEARNING MODEL")
		elif chosen_number > 0:
			print("### START WITH THE SELECTED MODEL: ", end="")
			d = torch.load(model_file_dict[chosen_number])
			self.model_for_judge_q.load_state_dict(d['model'])
