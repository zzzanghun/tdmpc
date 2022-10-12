import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'  #glfw
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger
from collections import deque
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		if video: video.init(env, enabled=(i==0))
		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			ep_reward += reward
			if video: video.record(env)
			t += 1
		episode_rewards.append(ep_reward)
		if video: video.save(env_step)
	return np.nanmean(episode_rewards)


def train(cfg):
	"""Training script for TD-MPC. Requires a CUDA-enabled device."""
	# assert torch.cuda.is_available()
	set_seed(cfg.seed)
	total_train_episode_results = []
	total_test_episode_results = []
	int_reward_deque = deque(maxlen=cfg.episode_length)
	work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	
	# Run training
	for train_idx in range(cfg.num_train):
		train_episode_results = []
		test_episode_results = []
		env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)
		L = logger.Logger(work_dir, cfg, str(cfg.server)+"_"+cfg.name_for_result_save+"_{}".format(train_idx), str(cfg.server)+"_"+cfg.name_for_result_save, train_idx)
		episode_idx, start_time = 0, time.time()
		for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):

			# Collect trajectory
			obs = env.reset()
			episode = Episode(cfg, obs)
			while not episode.done:
				action = agent.plan(obs, step=step, t0=episode.first)
				obs, reward, done, _ = env.step(action.cpu().numpy())
				if cfg.CURIOSITY_DRIVEN_EXPLORATION:
					int_reward = agent.calc_int_reward(obs, action)
					int_reward_deque.append(int_reward)
					reward = reward + int_reward
				episode += (obs, action, reward, done)
			if cfg.CURIOSITY_DRIVEN_EXPLORATION:
				assert len(episode) == cfg.episode_length and len(int_reward_deque) == cfg.episode_length
			else:
				assert len(episode) == cfg.episode_length
			buffer += episode

			# Update model
			train_metrics = {}
			if step >= cfg.seed_steps:
				num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
				for i in range(num_updates):
					train_metrics.update(agent.update(buffer, step+i))

			# Log training episode
			episode_idx += 1
			env_step = int(step*cfg.action_repeat)
			if cfg.CURIOSITY_ENCODER_FLAG_TIME and env_step > 100000:
				cfg.CURIOSITY_ENCODER = False
				cfg.CURIOSITY_DRIVEN_EXPLORATION = False
				cfg.CURIOSITY_ENCODER_FLAG_TIME = False
			if cfg.CURIOSITY_DRIVEN_EXPLORATION:
				common_metrics = {
					'episode': episode_idx,
					'step': step,
					'env_step': env_step,
					'total_time': time.time() - start_time,
					'episode_reward': episode.cumulative_reward - sum(int_reward_deque),
					'intrinsic_reward': sum(int_reward_deque),
					'episode_reward + intrinsic_reward': episode.cumulative_reward
				}
			else:
				common_metrics = {
					'episode': episode_idx,
					'step': step,
					'env_step': env_step,
					'total_time': time.time() - start_time,
					'episode_reward': episode.cumulative_reward
				}
			train_metrics.update(common_metrics)
			L.log(train_metrics, category='train')

			train_episode_results.append(float(episode.cumulative_reward))

			# Evaluate agent periodically
			if env_step % cfg.eval_freq == 0:
				test_episode_rewards = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
				common_metrics['episode_reward'] = test_episode_rewards
				test_episode_results.append(test_episode_rewards)
				L.log(common_metrics, category='eval')

		L.finish(agent)
		total_train_episode_results.append(train_episode_results)
		total_test_episode_results.append(test_episode_results)
		print('{}_Training completed successfully'.format(train_idx + 1))
	assert len(total_train_episode_results) == cfg.num_train
	assert len(total_test_episode_results) == cfg.num_train
	train_mean_results, train_max_results, train_min_results = logger.extract_results(total_train_episode_results)
	logger.graph_results(
		mean_results=train_mean_results, max_results=train_max_results, min_results=train_min_results, cfg=cfg, mode="Train"
	)
	logger.save_csv(
		results=total_train_episode_results, cfg=cfg, mode="Train"
	)

	test_mean_results, test_max_results, test_min_results = logger.extract_results(total_test_episode_results)
	logger.graph_results(
		mean_results=test_mean_results, max_results=test_max_results, min_results=test_min_results, cfg=cfg,
		mode="Test"
	)
	logger.save_csv(
		results=total_test_episode_results, cfg=cfg, mode="Test"
	)
	print("ALL TRAINING COMPLETED, SAVE GRAPH AND CSV")


if __name__ == '__main__':
	train_result = train(parse_cfg(Path().cwd() / __CONFIG__))
