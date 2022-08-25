import sys
import os
import datetime
import re
import numpy as np
import torch
import pandas as pd
from termcolor import colored
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


CONSOLE_FORMAT = [('episode', 'E', 'int'), ('env_step', 'S', 'int'), ('episode_reward', 'R', 'float'), ('total_time', 'T', 'time')]
AGENT_METRICS = ['consistency_loss', 'reward_loss', 'value_loss', 'total_loss', 'weighted_loss', 'pi_loss', 'grad_norm']
PROJECT_HOME = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'results')


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg, reward=None):
	"""Pretty-printing of run information. Call at start of training."""
	prefix, color, attrs = '  ', 'green', ['bold']
	def limstr(s, maxlen=32):
		return str(s[:maxlen]) + '...' if len(str(s)) > maxlen else s
	def pprint(k, v):
		print(prefix + colored(f'{k.capitalize()+":":<16}', color, attrs=attrs), limstr(v))
	kvs = [('task', cfg.task_title),
		   ('train steps', f'{int(cfg.train_steps*cfg.action_repeat):,}'),
		   ('observations', 'x'.join([str(s) for s in cfg.obs_shape])),
		   ('actions', cfg.action_dim),
		   ('experiment', cfg.exp_name)]
	if reward is not None:
		kvs.append(('episode reward', colored(str(int(reward)), 'white', attrs=['bold'])))
	w = np.max([len(limstr(str(kv[1]))) for kv in kvs]) + 21
	div = '-'*w
	print(div)
	for k,v in kvs:
		pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""Return a wandb-safe group name for logging. Optionally returns group name as list."""
	lst = [cfg.task, cfg.modality, re.sub('[^0-9a-zA-Z]+', '-', cfg.exp_name)]
	return lst if return_list else '-'.join(lst)


class VideoRecorder:
	"""Utility class for logging evaluation videos."""
	def __init__(self, root_dir, wandb, render_size=384, fps=15):
		self.save_dir = (root_dir / 'eval_video') if root_dir else None
		self._wandb = wandb
		self.render_size = render_size
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self.save_dir and self._wandb and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			frame = env.render(mode='rgb_array', height=self.render_size, width=self.render_size, camera_id=0)
			self.frames.append(frame)

	def save(self, step):
		if self.enabled:
			frames = np.stack(self.frames).transpose(0, 3, 1, 2)
			self._wandb.log({'eval_video': self._wandb.Video(frames, fps=self.fps, format='mp4')}, step=step)


class Logger(object):
	"""Primary logger object. Logs either locally or using wandb."""
	def __init__(self, log_dir, cfg, name, name_wandb, train_idx):
		self._log_dir = make_dir(log_dir)
		self._model_dir = make_dir(self._log_dir / 'models')
		self._save_model = cfg.save_model
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._cfg = cfg
		self._eval = []
		self._name = name
		self._name_wandb = name_wandb
		self._train_idx = train_idx
		print_run(cfg)
		project, entity = cfg.get('wandb_project', 'none'), cfg.get('wandb_entity', 'none')
		run_offline = not cfg.get('use_wandb', False) or project == 'none' or entity == 'none'
		if run_offline:
			print(colored('Logs will be saved locally.', 'yellow', attrs=['bold']))
			self._wandb = None
		else:
			try:
				os.environ["WANDB_SILENT"] = "true"
				import wandb
				wandb.init(project=project,
						entity=entity,
						name=self._name_wandb,
						group=self._group,
						tags=cfg_to_group(cfg, return_list=True) + [f'seed:{cfg.seed}'],
						dir=self._log_dir,
						config=OmegaConf.to_container(cfg, resolve=True))
				print(colored('Logs will be synced with wandb.', 'blue', attrs=['bold']))
				self._wandb = wandb
			except:
				print(colored('Warning: failed to init wandb. Logs will be saved locally.', 'yellow'), attrs=['bold'])
				self._wandb = None
		self._video = VideoRecorder(log_dir, self._wandb) if self._wandb and cfg.save_video else None

	@property
	def video(self):
		return self._video

	def finish(self, agent):
		now = datetime.datetime.now()
		local_now = now.astimezone()
		model_save_dir = os.path.join(PROJECT_HOME, self._cfg.task, self._cfg.name_for_result_save, "{}_{}".format(
			local_now.month, local_now.day), 'model')
		if not os.path.exists(model_save_dir):
			os.makedirs(model_save_dir, exist_ok=True)
		if self._save_model:
			fp = self._model_dir / f'model.pt'
			torch.save(agent.state_dict(), os.path.join(model_save_dir, 'model_{}.pth'.format(self._train_idx)))
			# if self._wandb:
			# 	artifact = self._wandb.Artifact(self._group+'-'+str(self._seed), type='model')
			# 	artifact.add_file(fp)
			# 	self._wandb.log_artifact(artifact)
		if self._wandb:
			self._wandb.finish()
		print_run(self._cfg, self._eval[-1][-1])

	def _format(self, key, value, ty):
		if ty == 'int':
			return f'{colored(key+":", "grey")} {int(value):,}'
		elif ty == 'float':
			return f'{colored(key+":", "grey")} {value:.01f}'
		elif ty == 'time':
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "grey")} {value}'
		else:
			raise f'invalid log format type: {ty}'

	def _print(self, d, category):
		category = colored(category, 'blue' if category == 'train' else 'green')
		pieces = [f' {category:<14}']
		for k, disp_k, ty in CONSOLE_FORMAT:
			pieces.append(f'{self._format(disp_k, d.get(k, 0), ty):<26}')
		print('   '.join(pieces))

	def log(self, d, category='train'):
		assert category in {'train', 'eval'}
		if self._wandb is not None:
			for k,v in d.items():
				self._wandb.log({category + '/' + k: v}, step=d['env_step'])
		if category == 'eval':
			keys = ['env_step', 'episode_reward']
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(self._log_dir / 'eval.log', header=keys, index=None)
		self._print(d, category)


def extract_results(episode_results):
	if not type(episode_results) == np.ndarray:
		episode_results = np.asarray(episode_results)

	mean_results = episode_results.mean(axis=0)
	assert len(mean_results) == len(episode_results[0])
	max_results = episode_results.max(axis=0)
	assert len(max_results) == len(episode_results[0])
	min_results = episode_results.min(axis=0)
	assert len(min_results) == len(episode_results[0])

	return mean_results, max_results, min_results


def graph_results(mean_results, max_results, min_results, cfg, mode):
	now = datetime.datetime.now()
	local_now = now.astimezone()

	graph_save_dir = os.path.join(PROJECT_HOME, cfg.task, cfg.name_for_result_save, "{}_{}_{}_{}".format(
			local_now.month, local_now.day, local_now.hour, local_now.minute), 'graph')
	if not os.path.exists(graph_save_dir):
		os.makedirs(graph_save_dir, exist_ok=True)

	plt.figure(figsize=(12, 5))
	plt.plot(
		[i for i in range(len(mean_results))],
		# mean_1[:len(step)],
		mean_results
	)
	plt.fill_between(
		[i for i in range(len(mean_results))],
		# min_1[:len(step)],
		# max_1[:len(step)],
		min_results,
		max_results,
		alpha=0.2
	)
	plt.title(''.format(cfg.name_for_result_save), fontsize=30)
	plt.ylabel("Train Episode Reward", fontsize=18)
	plt.xlabel("Num Episode", fontsize=18)
	plt.legend(loc=2, fancybox=True, framealpha=0.3)
	# plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
	plt.savefig("{0}.png".format(
		os.path.join(graph_save_dir, mode)), bbox_inches='tight')


def save_csv(mean_results, max_results, min_results, cfg, mode):
	now = datetime.datetime.now()
	local_now = now.astimezone()

	csv_save_dir = os.path.join(PROJECT_HOME, cfg.task, cfg.name_for_result_save, "{}_{}_{}_{}".format(
			local_now.month, local_now.day, local_now.hour, local_now.minute), 'csv')
	if not os.path.exists(csv_save_dir):
		os.makedirs(csv_save_dir, exist_ok=True)

	dict_for_save_csv = {'MEAN': mean_results, 'MAX': max_results, 'MIN': min_results}
	pd_data_frame = pd.DataFrame(dict_for_save_csv)
	pd_data_frame.to_csv("{0}.csv".format(os.path.join(csv_save_dir, mode)))
