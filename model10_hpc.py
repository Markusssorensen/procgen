import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms 
import random
import numpy as np
import pandas as pd
import math
from projectutils import make_env, Storage, orthogonal_init, BaseStorage
from modelutils import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import os

#Video and weights name
pathname = 'D:/OneDrive - Danmarks Tekniske Universitet/Studie/5. Semester/Deep Learning/project/'
# pathname = "/zhome/69/1/137385/Desktop/DeepLearning/ProjectWork/procgen/"
dirname = "model10"

name = "/CR_" + dirname
total_path = pathname + dirname + name

#Env hyperparams
n_envs=32
env_name='coinrun'
num_levels=500
start_level= 0
use_backgrounds=True
normalize_obs=False
normalize_reward=False
seed=0

#Test env
test_n_envs=25
test_env_name='coinrun'
test_num_levels=50
n_eval_levels = 1000
test_start_level= start_level + num_levels + 1
test_use_backgrounds=True
test_normalize_obs=False
test_normalize_reward=False
test_seed=0
 
#Train Test hyperparams
reward_dieing = -1
total_steps = 15000000
num_steps = 256
num_epochs = 2
batch_size = 256
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = 0.01

##Policy hyperparams
img_height = 64
img_width = 64
enc_inp_layers = 3
enc_out_dim = 512
n_obs_back = 5
n_heads = 4
transformer_dropout = 0.1
transformer_forward_scale = 4
transformer_out_dim = (1+n_obs_back)*enc_out_dim
obs_mask = torch.triu(torch.ones(n_obs_back+1,n_obs_back+1)).cuda()

## Augmentation params
brightness = 0.5
p_bright = 0.5
contrast = 0.5
p_contr = 0.4
saturation = 0.3
p_satur = 0.4
hue = 0.5
p_hue = 0.5
augment_prob = 0.85

#Optimizer hyperparams
opt_lr = 5e-4
opt_eps = 1e-5

env = make_env(n_envs=n_envs, 
               env_name=env_name,
               num_levels=num_levels,
               start_level= start_level,
               use_backgrounds=use_backgrounds,
               normalize_obs=normalize_obs,
               normalize_reward=normalize_reward,
               seed=seed
               )

print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)

class Policy6(nn.Module):
  def __init__(self, encoder, pos_encoder_obs, transformer_block_obs, transformer_out_dim , num_actions):
    super().__init__()
    self.encoder = encoder
    self.pos_encoder_obs = pos_encoder_obs
    self.transformer_block_obs = transformer_block_obs
    # self.linear1 = orthogonal_init(nn.Linear(transformer_out_dim, int(transformer_out_dim/2)), gain=.01)
    # self.linear2 = orthogonal_init(nn.Linear(int(transformer_out_dim/2), int(transformer_out_dim/4)), gain=.01)
    # self.policy = orthogonal_init(nn.Linear(int(transformer_out_dim/4), num_actions), gain=.01)
    # self.value = orthogonal_init(nn.Linear(int(transformer_out_dim/4), 1), gain=1.)
    self.policy = orthogonal_init(nn.Linear(int(transformer_out_dim), num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(int(transformer_out_dim), 1), gain=1.)

  def act(self, x, obs_back, obs_mask = None):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x, obs_back, obs_mask)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x, obs_back, obs_mask = None):
    all_obs = list()
    x = self.encoder(x)
    all_obs.append(x)
    obs_back = torch.transpose(obs_back,0,1)
    for i in obs_back:
        i = encoder(i)
        all_obs.append(i)
    
    x = torch.stack(all_obs,1)

    x = self.pos_encoder_obs(x)
    
    x = self.transformer_block_obs(x,x,x,obs_mask)
    
    x = x.view(x.size(0), -1)
    
    # x = F.relu(self.linear1(x))
    # x = F.relu(self.linear2(x))
    
    logits = F.softmax(self.policy(x),dim=1)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value


# Define environment
# check the utils.py file for info on arguments

encoder = BaselineEncoder(enc_inp_layers,enc_out_dim).cuda()
pos_encoder = PositionalEncoder(enc_out_dim,n_obs_back+1).cuda()
attention_obs = MultiHeadAttention(n_heads,enc_out_dim).cuda()
transformer_obs = TransformerBlock(attention_obs,enc_out_dim,transformer_dropout,transformer_forward_scale).cuda()
policy = Policy6(encoder, pos_encoder, transformer_obs, transformer_out_dim, env.action_space.n)
policy = policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=opt_lr, eps=opt_eps)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    n_envs,
    n_obs_back,
    env.action_space.n
)



step_ns = []
mean_rewards = []
time_lst = []

# Run training
obs = env.reset()
step = 0

time0 = time.time()
while step < total_steps:

  # Use policy to collect data for "num_steps" steps in the environment. 
  policy.eval()
  prev_actions = torch.zeros([n_envs,n_obs_back, env.action_space.n])
  prev_actions2 = torch.zeros([n_envs,n_obs_back, env.action_space.n])
  prev_obs = torch.zeros([n_envs,n_obs_back, 3, img_height, img_width])
  prev_obs2 = torch.zeros([n_envs,n_obs_back, 3, img_height, img_width])
  
  for _ in range(num_steps):
    # Use policy
    obs = data_augmentation(obs)
    
    action, log_prob, value = policy.act(obs.cuda(),prev_obs.cuda(),obs_mask.cuda()) #add prev_actions.cuda() to act if last actions are used in policy
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)
    
    done_idx = [i for i, e in enumerate(done) if e == 1]
    for i in done_idx:
        if int(reward[i]) == 0:
            reward[i] = reward_dieing
    
    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value, prev_actions, prev_obs) #add ", prev_actions, prev_obs" if prev actions are used

    prev_obs2[:,1:] = prev_obs[:,:n_obs_back-1].clone()
    prev_obs2[:,0] = obs.clone()
    prev_obs = prev_obs2.clone()
    prev_obs[done,:,:] = 0
    
    # Update current observation
    obs = next_obs
    
    #Update prev_actions
    prev_actions2[:,1:,:] = prev_actions[:,:n_obs_back-1,:].clone()
    prev_actions2[:,0,:] = nn.functional.one_hot(action, num_classes=15)
    prev_actions = prev_actions2.clone()
    prev_actions[done,:,:] = 0
    

  # Add the last observation to collected data
  _, _, value = policy.act(obs.cuda(),prev_obs.cuda(),obs_mask.cuda()) # add prev_actions.cuda() if prev actions are used
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # Optimize policy
  policy.train()
  
  for epoch in range(num_epochs):

    # Iterate over batches of transitions
    generator = storage.get_generator(batch_size)
    for batch in generator:
      b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage, b_prev_obs = batch # add b_prev_actions if used

      # Get current policy outputs
      new_dist, new_value = policy(b_obs, b_prev_obs, obs_mask.cuda()) # add b_prev_actions if prev actions are used
      new_log_prob = new_dist.log_prob(b_action)

      # Clipped policy objective
      ratio = torch.exp(new_log_prob - b_log_prob)
      clipped_ratio = ratio.clamp(min=1-eps, max=1+eps)
      pi_loss = torch.min(ratio*b_advantage, clipped_ratio*b_advantage)
      pi_loss = pi_loss.mean()

      # Clipped value function objective
      value_diff = new_value - b_value
      clipped_value = b_value + value_diff.clamp(min=-eps,max=eps)
      value_loss = torch.max((clipped_value - b_returns) ** 2, (new_value - b_returns) ** 2)
      value_loss = value_loss.mean()
      
      # Entropy loss
      entropy_loss = new_dist.entropy().mean()

      # Backpropagate losses
      loss = - pi_loss + value_coef * value_loss - entropy_coef * entropy_loss
      loss.backward() #Måske ikke mean? sum? eller?

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

      # Update policy
      optimizer.step()
      optimizer.zero_grad()

  # Update stats
  step += n_envs * num_steps
  print(f'Step: {step}\tMean reward: {storage.get_reward()}')
  step_ns.append(step)
  mean_rewards.append(storage.get_reward())
  time_lst.append(time.time() - time0)
  if step % 500000 < 10000:
      torch.save(policy.state_dict(), total_path + '.pt')  
      train_df = pd.DataFrame({'Training Steps': step_ns, 'Mean Reward': mean_rewards, 'Time elapsed': time_lst})
      train_df.to_csv(total_path + '_train.csv')
  
time1 = time.time()
total_time = time1-time0
print('Completed training! \nTime used: ' + str(total_time))

train_df = pd.DataFrame({'Training Steps': step_ns, 'Mean Reward': mean_rewards, 'Time elapsed': time_lst})

train_df.to_csv(total_path + '_train.csv')

# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(step_ns,mean_rewards)
# plt.xlabel("Number of steps")
# plt.ylabel("Average Reward gained")
# plt.title("CR " + dirname)
# plt.grid(True)
# plt.savefig(total_path + '.pdf')

torch.save(policy.state_dict(), total_path + '.pt')

# import imageio

# # Make evaluation environment
# eval_env = make_env(n_envs=test_n_envs, 
#                env_name=test_env_name,
#                num_levels=test_num_levels,
#                start_level= test_start_level,
#                use_backgrounds=test_use_backgrounds,
#                normalize_obs=test_normalize_obs,
#                normalize_reward=test_normalize_reward,
#                seed=test_seed
#                )

# obs = eval_env.reset()

# frames = []
# total_reward = []

# # Evaluate policy
# policy.eval()
# done_vec = torch.tensor([0 for i in range(test_n_envs)])

# # test_prev_actions = torch.zeros([test_n_envs,n_action_back, env.action_space.n])
# # test_prev_actions2 = torch.zeros([test_n_envs,n_action_back, env.action_space.n])

# n_steps_levels = np.zeros([test_n_envs, int(n_eval_levels/test_n_envs)])
# reward_levels = np.zeros([test_n_envs, int(n_eval_levels/test_n_envs)])

# n_test_steps = 0
# while min(done_vec) < int(n_eval_levels/test_n_envs):
  
#   # Use policy
#   action, log_prob, value = policy.act(obs.cuda()) #Add , test_prev_actions.cuda() if prev actions are used

#   # Take step in environment
#   obs, reward, done, info = eval_env.step(action)
#   total_reward.append(torch.Tensor(reward))      
  
#   if done_vec[0] < int(n_eval_levels/test_n_envs):
#       # Render environment and store
#       frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
#       frames.append(frame)
  
#   done_idx = [i for i, e in enumerate(done) if e == 1]
#   for i in done_idx:
#       if int(reward[i]) == 0:
#           reward[i] = reward_dieing
#       if not done_vec[i] > int(n_eval_levels/test_n_envs)-1:    
#           reward_levels[i,done_vec[i]] = reward[i]
#           if done_vec[i] > 0:
#               n_steps_levels[i,done_vec[i]] = n_test_steps - sum(n_steps_levels[i,:done_vec[i]])
#           else:
#               n_steps_levels[i,done_vec[i]] = n_test_steps
  
#   done_vec = done_vec + done
#   #Update prev_actions
#   # test_prev_actions2[:,1:,:] = test_prev_actions[:,:n_action_back-1,:]
#   # test_prev_actions2[:,0,:] = nn.functional.one_hot(action, num_classes=15)
#   # test_prev_actions = test_prev_actions2
#   # test_prev_actions[done,:,:] = 0
#   n_test_steps += 1
  
#   if n_test_steps % 1000 == 0:
#       print(done_vec)

# # Calculate average return
# n_steps_levels = np.reshape(n_steps_levels,(1,n_steps_levels.shape[0]*n_steps_levels.shape[1]))
# reward_levels = np.reshape(reward_levels,(1,reward_levels.shape[0]*reward_levels.shape[1]))

# eval_df = pd.DataFrame({'Level':range(n_steps_levels.shape[1]), 'N Steps':n_steps_levels[0,:], 'Reward':reward_levels[0,:], 'Training duration (s)':[total_time for i in range(n_steps_levels.shape[1])]})

# eval_df.to_csv(total_path + '_test.csv')

# # Save frames as video
# frames = torch.stack(frames)
# imageio.mimsave(total_path + '2' + '.mp4', frames, fps=15) 