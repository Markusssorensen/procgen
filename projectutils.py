# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:29:10 2020

@author: Markus
"""

import contextlib
import os
from abc import ABC, abstractmethod
import numpy as np
import gym
import random
from gym import spaces
import time
from collections import deque
import os
import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
from procgen import ProcgenEnv
from collections import deque

"""
Utility functions for the deep RL projects that I supervise in 02456 Deep Learning @ DTU.
"""


def make_env(
	n_envs=32,
	env_name='starpilot',
	start_level=0,
	num_levels=100,
	use_backgrounds=False,
	normalize_obs=False,
	normalize_reward=True,
	seed=0
	):
	"""Make environment for procgen experiments"""
	set_global_seeds(seed)
	set_global_log_levels(40)
	env = ProcgenEnv(
		num_envs=n_envs,
		env_name=env_name,
		start_level=start_level,
		num_levels=num_levels,
		distribution_mode='easy',
		use_backgrounds=use_backgrounds,
		restrict_themes=not use_backgrounds,
		render_mode='rgb_array',
		rand_seed=seed
	)
	env = VecExtractDictObs(env, "rgb")
	env = VecNormalize(env, ob=normalize_obs, ret=normalize_reward)
	env = TransposeFrame(env)
	env = ScaledFloatFrame(env)
	env = TensorEnv(env)
	
	return env

class BaseStorage2():
    def __init__(self, obs_shape, num_steps, num_envs, gamma=0.99, lmbda=0.95, normalize_advantage=True):
        self.obs_shape = obs_shape
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.gamma = gamma
        self.lmbda = lmbda
        self.normalize_advantage = normalize_advantage
        self.reset()

    def reset(self):
        self.obs = list()
        self.action = list()
        self.reward = list()
        self.done = list()
        self.log_prob = list()
        self.value = list()
        self.returns = list()
        self.advantage = list()
        self.info = list()
        self.step = 0

    def store(self, obs, action, reward, done, info, log_prob, value):
        self.obs.append(obs.clone())
        self.action.append(action.clone())
        self.reward.append(torch.from_numpy(reward.copy()))
        self.done.append(torch.from_numpy(done.copy()))
        self.info.append(info)
        self.log_prob.append(log_prob.clone())
        self.value.append(value.clone())
        
    def store_last(self, obs, value):
        self.obs.append(obs.clone())
        self.value.append(value.clone())
        self.obs = torch.stack(self.obs,dim=0)
        self.action = torch.stack(self.action,dim=0)
        self.reward = torch.stack(self.reward,dim=0)
        self.done = torch.stack(self.done,dim=0)
        self.log_prob = torch.stack(self.log_prob,dim=0)
        self.value = torch.stack(self.value,dim=0)
        self.advantage = torch.zeros(self.num_steps, self.num_envs)

    def compute_return_advantage(self):
        advantage = 0
        for i in reversed(range(self.num_steps)):
            delta = (self.reward[i] + self.gamma * self.value[i+1] * (torch.logical_not(self.done[i]))) - self.value[i]
            advantage = self.gamma * self.lmbda * advantage * (torch.logical_not(self.done[i])) + delta
            self.advantage[i] = advantage

        self.returns = self.advantage + self.value[:-1]
        if self.normalize_advantage:
            self.advantage = (self.advantage - self.advantage.mean()) / (self.advantage.std() + 1e-9)

    def get_generator(self, batch_size=1024):
        iterator = BatchSampler(SubsetRandomSampler(range(self.num_steps*self.num_envs)), batch_size, drop_last=True)
        for indices in iterator:
            obs = self.obs[:-1].reshape(-1, *self.obs_shape)[indices].cuda()
            action = self.action.reshape(-1)[indices].cuda()
            log_prob = self.log_prob.reshape(-1)[indices].cuda()
            value = self.value[:-1].reshape(-1)[indices].cuda()
            returns = self.returns.reshape(-1)[indices].cuda()
            advantage = self.advantage.reshape(-1)[indices].cuda()
            yield obs, action, log_prob, value, returns, advantage

    def get_reward(self, normalized_reward=True):
        if normalized_reward:
            reward = []
            for step in range(self.num_steps):
                info = self.info[step]
                reward.append([d['reward'] for d in info])
            reward = torch.Tensor(reward)
        else:
            reward = self.reward
		
        return reward.mean(1).sum(0)

class BaseStorage():
	def __init__(self, obs_shape, num_steps, num_envs, gamma=0.99, lmbda=0.95, normalize_advantage=True):
		self.obs_shape = obs_shape
		self.num_steps = num_steps
		self.num_envs = num_envs
		self.gamma = gamma
		self.lmbda = lmbda
		self.normalize_advantage = normalize_advantage
		self.reset()

	def reset(self):
		self.obs = torch.zeros(self.num_steps+1, self.num_envs, *self.obs_shape)
		self.action = torch.zeros(self.num_steps, self.num_envs)
		self.reward = torch.zeros(self.num_steps, self.num_envs)
		self.done = torch.zeros(self.num_steps, self.num_envs)
		self.log_prob = torch.zeros(self.num_steps, self.num_envs)
		self.value = torch.zeros(self.num_steps+1, self.num_envs)
		self.returns = torch.zeros(self.num_steps, self.num_envs)
		self.advantage = torch.zeros(self.num_steps, self.num_envs)
		self.info = deque(maxlen=self.num_steps)
		self.step = 0

	def store(self, obs, action, reward, done, info, log_prob, value):
		self.obs[self.step] = obs.clone()
		self.action[self.step] = action.clone()
		self.reward[self.step] = torch.from_numpy(reward.copy())
		self.done[self.step] = torch.from_numpy(done.copy())
		self.info.append(info)
		self.log_prob[self.step] = log_prob.clone()
		self.value[self.step] = value.clone()
		self.step = (self.step + 1) % self.num_steps

	def store_last(self, obs, value):
		self.obs[-1] = obs.clone()
		self.value[-1] = value.clone()

	def compute_return_advantage(self):
		advantage = 0
		for i in reversed(range(self.num_steps)):
			delta = (self.reward[i] + self.gamma * self.value[i+1] * (1 - self.done[i])) - self.value[i]
			advantage = self.gamma * self.lmbda * advantage * (1 - self.done[i]) + delta
			self.advantage[i] = advantage

		self.returns = self.advantage + self.value[:-1]
		if self.normalize_advantage:
			self.advantage = (self.advantage - self.advantage.mean()) / (self.advantage.std() + 1e-9)

	def get_generator(self, batch_size=1024):
		iterator = BatchSampler(SubsetRandomSampler(range(self.num_steps*self.num_envs)), batch_size, drop_last=True)
		for indices in iterator:
			obs = self.obs[:-1].reshape(-1, *self.obs_shape)[indices].cuda()
			action = self.action.reshape(-1)[indices].cuda()
			log_prob = self.log_prob.reshape(-1)[indices].cuda()
			value = self.value[:-1].reshape(-1)[indices].cuda()
			returns = self.returns.reshape(-1)[indices].cuda()
			advantage = self.advantage.reshape(-1)[indices].cuda()
			yield obs, action, log_prob, value, returns, advantage

	def get_reward(self, normalized_reward=True):
		if normalized_reward:
			reward = []
			for step in range(self.num_steps):
				info = self.info[step]
				reward.append([d['reward'] for d in info])
			reward = torch.Tensor(reward)
		else:
			reward = self.reward
		
		return reward.mean(1).sum(0)


class Storage():
    def __init__(self, obs_shape, num_steps, num_envs, actions_back, n_actionsspace, gamma=0.99, lmbda=0.95, normalize_advantage=True):
        self.obs_shape = obs_shape
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.actions_back = actions_back
        self.n_actionsspace = n_actionsspace
        self.gamma = gamma
        self.lmbda = lmbda
        self.normalize_advantage = normalize_advantage
        self.reset()

    def reset(self):
        self.obs = torch.zeros(self.num_steps+1, self.num_envs, *self.obs_shape)
        self.action = torch.zeros(self.num_steps, self.num_envs)
        self.reward = torch.zeros(self.num_steps, self.num_envs)
        self.done = torch.zeros(self.num_steps, self.num_envs)
        self.log_prob = torch.zeros(self.num_steps, self.num_envs)
        self.value = torch.zeros(self.num_steps+1, self.num_envs)
        self.returns = torch.zeros(self.num_steps, self.num_envs)
        self.advantage = torch.zeros(self.num_steps, self.num_envs)
        self.info = deque(maxlen=self.num_steps)
        self.prev_actions = torch.zeros(self.num_steps, self.num_envs, self.actions_back, self.n_actionsspace)
        self.prev_obs = torch.zeros(self.num_steps, self.num_envs, self.actions_back, *self.obs_shape)
        self.step = 0

    def store(self, obs, action, reward, done, info, log_prob, value, prev_actions, prev_obs):
        self.obs[self.step] = obs.clone()
        self.prev_obs[self.step] = prev_obs.clone()
        self.action[self.step] = action.clone()
        self.reward[self.step] = torch.from_numpy(reward.copy())
        self.done[self.step] = torch.from_numpy(done.copy())
        self.info.append(info)
        self.log_prob[self.step] = log_prob.clone()
        self.value[self.step] = value.clone()
        self.step = (self.step + 1) % self.num_steps

    def store_last(self, obs, value):
        self.obs[-1] = obs.clone()
        self.value[-1] = value.clone()

    def compute_return_advantage(self):
        advantage = 0
        for i in reversed(range(self.num_steps)):
            delta = (self.reward[i] + self.gamma * self.value[i+1] * (1 - self.done[i])) - self.value[i]
            advantage = self.gamma * self.lmbda * advantage * (1 - self.done[i]) + delta
            self.advantage[i] = advantage

        self.returns = self.advantage + self.value[:-1]
        if self.normalize_advantage:
            self.advantage = (self.advantage - self.advantage.mean()) / (self.advantage.std() + 1e-9)

    def get_generator(self, batch_size=1024):
        iterator = BatchSampler(SubsetRandomSampler(range(self.num_steps*self.num_envs)), batch_size, drop_last=True)
        for indices in iterator:
            obs = self.obs[:-1].reshape(-1, *self.obs_shape)[indices].cuda()
            action = self.action.reshape(-1)[indices].cuda()
            log_prob = self.log_prob.reshape(-1)[indices].cuda()
            value = self.value[:-1].reshape(-1)[indices].cuda()
            returns = self.returns.reshape(-1)[indices].cuda()
            advantage = self.advantage.reshape(-1)[indices].cuda()
            prev_obs = self.prev_obs.reshape(-1,self.actions_back, *self.obs_shape)[indices].cuda()
            yield obs, action, log_prob, value, returns, advantage, prev_obs

    def get_reward(self, normalized_reward=True):
        if normalized_reward:
            reward = []
            for step in range(self.num_steps):
                info = self.info[step]
                reward.append([d['reward'] for d in info])
            reward = torch.Tensor(reward)
        else:
            reward = self.reward
		
        return reward.mean(1).sum(0)

# 	def compute_prev_actions(self, n_actionsspace, n_prev=10):
#         self.prev_action = torch.zeros([self.num_steps, self.num_envs, n_prev, n_actionsspace])
#         onehot_actions = nn.functional.one_hot(self.action, num_classes=n_actionsspace)
#         for i in self.action:
#             done_mask = torch.zeros(self.num_envs) == 0
#             for j in range(1, n_prev+1):
#                 if not (i-j < 0):
#                     done_mask = (self.done[i-j] == 0) * done_mask
#                     self.prev_action[i,done_mask,j,:] = onehot_actions[i-j,done_mask,:]

#         return
                    
                

class Storage2():
	def __init__(self, obs_shape, num_steps, num_envs, gamma=0.99, lmbda=0.95, normalize_advantage=True):
		self.obs_shape = obs_shape
		self.num_steps = num_steps
		self.num_envs = num_envs
		self.gamma = gamma
		self.lmbda = lmbda
		self.normalize_advantage = normalize_advantage
		self.reset()

	def reset(self):
		self.obs = torch.zeros(self.num_steps+1, self.num_envs, *self.obs_shape)
		self.action = torch.zeros(self.num_steps, self.num_envs)
		self.reward = torch.zeros(self.num_steps, self.num_envs)
		self.done = torch.zeros(self.num_steps, self.num_envs)
		self.log_prob = torch.zeros(self.num_steps, self.num_envs)
		self.value = torch.zeros(self.num_steps+1, self.num_envs)
		self.returns = torch.zeros(self.num_steps, self.num_envs)
		self.advantage = torch.zeros(self.num_steps, self.num_envs)
		self.info = deque(maxlen=self.num_steps)
		self.step = 0

	def store(self, obs, action, reward, done, info, log_prob, value):
		self.obs[:,self.step] = obs.clone()
		self.action[:,self.step] = action.clone()
		self.reward[:,self.step] = torch.from_numpy(reward.copy())
		self.done[:,self.step] = torch.from_numpy(done.copy())
		self.info.append(info)
		self.log_prob[:,self.step] = log_prob.clone()
		self.value[:,self.step] = value.clone()
		self.step = (self.step + 1) % self.num_steps

	def store_last(self, obs, value):
		self.obs[-1] = obs.clone()
		self.value[-1] = value.clone()

	def compute_return_advantage(self):
		advantage = 0
		for i in reversed(range(self.num_steps)):
			delta = (self.reward[i] + self.gamma * self.value[i+1] * (1 - self.done[i])) - self.value[i]
			advantage = self.gamma * self.lmbda * advantage * (1 - self.done[i]) + delta
			self.advantage[i] = advantage

		self.returns = self.advantage + self.value[:-1]
		if self.normalize_advantage:
			self.advantage = (self.advantage - self.advantage.mean()) / (self.advantage.std() + 1e-9)

	def get_generator(self, batch_size=1024):
		iterator = BatchSampler(SubsetRandomSampler(range(self.num_steps*self.num_envs)), batch_size, drop_last=True)
		for indices in iterator:
			obs = self.obs[:-1].reshape(-1, *self.obs_shape)[indices].cuda()
			action = self.action.reshape(-1)[indices].cuda()
			log_prob = self.log_prob.reshape(-1)[indices].cuda()
			value = self.value[:-1].reshape(-1)[indices].cuda()
			returns = self.returns.reshape(-1)[indices].cuda()
			advantage = self.advantage.reshape(-1)[indices].cuda()
			yield obs, action, log_prob, value, returns, advantage

	def get_reward(self, normalized_reward=True):
		if normalized_reward:
			reward = []
			for step in range(self.num_steps):
				info = self.info[step]
				reward.append([d['reward'] for d in info])
			reward = torch.Tensor(reward)
		else:
			reward = self.reward
		
		return reward.mean(1).sum(0)



def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
	"""Orthogonal weight initialization: https://arxiv.org/abs/1312.6120"""
	if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
		nn.init.orthogonal_(module.weight.data, gain)
		nn.init.constant_(module.bias.data, 0)
	return module


"""
Helper functions that set global seeds and gym logging preferences
"""

def set_global_seeds(seed):
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def set_global_log_levels(level):
	gym.logger.set_level(level)


"""
Copy-pasted from OpenAI to obviate dependency on Baselines. Required for vectorized environments.
You will never have to look beyond this line.
"""

class AlreadySteppingError(Exception):
	"""
	Raised when an asynchronous step is running while
	step_async() is called again.
	"""

	def __init__(self):
		msg = 'already running an async step'
		Exception.__init__(self, msg)


class NotSteppingError(Exception):
	"""
	Raised when an asynchronous step is not running but
	step_wait() is called.
	"""

	def __init__(self):
		msg = 'not running an async step'
		Exception.__init__(self, msg)


class VecEnv(ABC):
	"""
	An abstract asynchronous, vectorized environment.
	Used to batch data from multiple copies of an environment, so that
	each observation becomes an batch of observations, and expected action is a batch of actions to
	be applied per-environment.
	"""
	closed = False
	viewer = None

	metadata = {
		'render.modes': ['human', 'rgb_array']
	}

	def __init__(self, num_envs, observation_space, action_space):
		self.num_envs = num_envs
		self.observation_space = observation_space
		self.action_space = action_space

	@abstractmethod
	def reset(self):
		"""
		Reset all the environments and return an array of
		observations, or a dict of observation arrays.

		If step_async is still doing work, that work will
		be cancelled and step_wait() should not be called
		until step_async() is invoked again.
		"""
		pass

	@abstractmethod
	def step_async(self, actions):
		"""
		Tell all the environments to start taking a step
		with the given actions.
		Call step_wait() to get the results of the step.

		You should not call this if a step_async run is
		already pending.
		"""
		pass

	@abstractmethod
	def step_wait(self):
		"""
		Wait for the step taken with step_async().

		Returns (obs, rews, dones, infos):
		 - obs: an array of observations, or a dict of
				arrays of observations.
		 - rews: an array of rewards
		 - dones: an array of "episode done" booleans
		 - infos: a sequence of info objects
		"""
		pass

	def close_extras(self):
		"""
		Clean up the  extra resources, beyond what's in this base class.
		Only runs when not self.closed.
		"""
		pass

	def close(self):
		if self.closed:
			return
		if self.viewer is not None:
			self.viewer.close()
		self.close_extras()
		self.closed = True

	def step(self, actions):
		"""
		Step the environments synchronously.

		This is available for backwards compatibility.
		"""
		self.step_async(actions)
		return self.step_wait()

	def render(self, mode='human'):
		imgs = self.get_images()
		bigimg = "ARGHH" #tile_images(imgs)
		if mode == 'human':
			self.get_viewer().imshow(bigimg)
			return self.get_viewer().isopen
		elif mode == 'rgb_array':
			return bigimg
		else:
			raise NotImplementedError

	def get_images(self):
		"""
		Return RGB images from each environment
		"""
		raise NotImplementedError

	@property
	def unwrapped(self):
		if isinstance(self, VecEnvWrapper):
			return self.venv.unwrapped
		else:
			return self

	def get_viewer(self):
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.SimpleImageViewer()
		return self.viewer

	
class VecEnvWrapper(VecEnv):
	"""
	An environment wrapper that applies to an entire batch
	of environments at once.
	"""

	def __init__(self, venv, observation_space=None, action_space=None):
		self.venv = venv
		super().__init__(num_envs=venv.num_envs,
						observation_space=observation_space or venv.observation_space,
						action_space=action_space or venv.action_space)

	def step_async(self, actions):
		self.venv.step_async(actions)

	@abstractmethod
	def reset(self):
		pass

	@abstractmethod
	def step_wait(self):
		pass

	def close(self):
		return self.venv.close()

	def render(self, mode='human'):
		return self.venv.render(mode=mode)

	def get_images(self):
		return self.venv.get_images()

	def __getattr__(self, name):
		if name.startswith('_'):
			raise AttributeError("attempted to get missing private attribute '{}'".format(name))
		return getattr(self.venv, name)

	
class VecEnvObservationWrapper(VecEnvWrapper):
	@abstractmethod
	def process(self, obs):
		pass

	def reset(self):
		obs = self.venv.reset()
		return self.process(obs)

	def step_wait(self):
		obs, rews, dones, infos = self.venv.step_wait()
		return self.process(obs), rews, dones, infos

	
class CloudpickleWrapper(object):
	"""
	Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
	"""

	def __init__(self, x):
		self.x = x

	def __getstate__(self):
		import cloudpickle
		return cloudpickle.dumps(self.x)

	def __setstate__(self, ob):
		import pickle
		self.x = pickle.loads(ob)

		
@contextlib.contextmanager
def clear_mpi_env_vars():
	"""
	from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
	This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing
	Processes.
	"""
	removed_environment = {}
	for k, v in list(os.environ.items()):
		for prefix in ['OMPI_', 'PMI_']:
			if k.startswith(prefix):
				removed_environment[k] = v
				del os.environ[k]
	try:
		yield
	finally:
		os.environ.update(removed_environment)

		
class VecFrameStack(VecEnvWrapper):
	def __init__(self, venv, nstack):
		self.venv = venv
		self.nstack = nstack
		wos = venv.observation_space  # wrapped ob space
		low = np.repeat(wos.low, self.nstack, axis=-1)
		high = np.repeat(wos.high, self.nstack, axis=-1)
		self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
		observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
		VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

	def step_wait(self):
		obs, rews, news, infos = self.venv.step_wait()
		self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
		for (i, new) in enumerate(news):
			if new:
				self.stackedobs[i] = 0
		self.stackedobs[..., -obs.shape[-1]:] = obs
		return self.stackedobs, rews, news, infos

	def reset(self):
		obs = self.venv.reset()
		self.stackedobs[...] = 0
		self.stackedobs[..., -obs.shape[-1]:] = obs
		return self.stackedobs
	
class VecExtractDictObs(VecEnvObservationWrapper):
	def __init__(self, venv, key):
		self.key = key
		super().__init__(venv=venv,
			observation_space=venv.observation_space.spaces[self.key])

	def process(self, obs):
		return obs[self.key]
	
	
class RunningMeanStd(object):
	# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
	def __init__(self, epsilon=1e-4, shape=()):
		self.mean = np.zeros(shape, 'float64')
		self.var = np.ones(shape, 'float64')
		self.count = epsilon

	def update(self, x):
		batch_mean = np.mean(x, axis=0)
		batch_var = np.var(x, axis=0)
		batch_count = x.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)

	def update_from_moments(self, batch_mean, batch_var, batch_count):
		self.mean, self.var, self.count = update_mean_var_count_from_moments(
			self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

		
def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
	delta = batch_mean - mean
	tot_count = count + batch_count

	new_mean = mean + delta * batch_count / tot_count
	m_a = var * count
	m_b = batch_var * batch_count
	M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
	new_var = M2 / tot_count
	new_count = tot_count

	return new_mean, new_var, new_count


class VecNormalize(VecEnvWrapper):
	"""
	A vectorized wrapper that normalizes the observations
	and returns from an environment.
	"""

	def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
		VecEnvWrapper.__init__(self, venv)

		self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
		self.ret_rms = RunningMeanStd(shape=()) if ret else None
		
		self.clipob = clipob
		self.cliprew = cliprew
		self.ret = np.zeros(self.num_envs)
		self.gamma = gamma
		self.epsilon = epsilon

	def step_wait(self):
		obs, rews, news, infos = self.venv.step_wait()
		for i in range(len(infos)):
			infos[i]['reward'] = rews[i]
		self.ret = self.ret * self.gamma + rews
		obs = self._obfilt(obs)
		if self.ret_rms:
			self.ret_rms.update(self.ret)
			rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
		self.ret[news] = 0.
		return obs, rews, news, infos

	def _obfilt(self, obs):
		if self.ob_rms:
			self.ob_rms.update(obs)
			obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
			return obs
		else:
			return obs

	def reset(self):
		self.ret = np.zeros(self.num_envs)
		obs = self.venv.reset()
		return self._obfilt(obs)


class TransposeFrame(VecEnvWrapper):
	def __init__(self, env):
		super().__init__(venv=env)
		obs_shape = self.observation_space.shape
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.float32)

	def step_wait(self):
		obs, reward, done, info = self.venv.step_wait()
		return obs.transpose(0,3,1,2), reward, done, info

	def reset(self):
		obs = self.venv.reset()
		return obs.transpose(0,3,1,2)


class ScaledFloatFrame(VecEnvWrapper):
	def __init__(self, env):
		super().__init__(venv=env)
		obs_shape = self.observation_space.shape
		self.observation_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

	def step_wait(self):
		obs, reward, done, info = self.venv.step_wait()
		return obs/255.0, reward, done, info

	def reset(self):
		obs = self.venv.reset()
		return obs/255.0


class TensorEnv(VecEnvWrapper):
	def __init__(self, env):
		super().__init__(venv=env)

	def step_async(self, actions):
		if isinstance(actions, torch.Tensor):
			actions = actions.detach().cpu().numpy()
		self.venv.step_async(actions)

	def step_wait(self):
		obs, reward, done, info = self.venv.step_wait()
		return torch.Tensor(obs), reward, done, info

	def reset(self):
		obs = self.venv.reset()
		return torch.Tensor(obs)