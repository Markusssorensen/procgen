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
import re

#Video and weights name
pathname = 'D:/OneDrive - Danmarks Tekniske Universitet/Studie/5. Semester/Deep Learning/project/'
dirname = 'model1'

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
reward_dieing = 0
total_steps = 15000000
num_steps = 256
num_epochs = 2
batch_size = 256
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = 0.01

#Policy hyperparams
in_channels = 3
feature_dim = 512


# ##Imagesplit
# img_height = 64
# img_width = img_height
# vert_splits = 4
# hor_splits = vert_splits
# ##Image Encoder
# img_inp_layers = 3
# enc_out_dim_img = 128
# ##Action Encoder
# act_in_features = 15
# act_l1_features = 128
# act_l2_features = 128
# act_out_features = 32
# n_action_back = 5
# ##Transformer Image
# img_heads = 4
# forward_scaling_img = 4
# transf_dropout1 = 0.1
# ##Transformer Action
# seq_heads = 2
# transf_dropout2 = 0.1
# forward_scaling_seq = 4
# policy_lin = enc_out_dim_img*vert_splits*hor_splits + n_action_back*act_out_features

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


# Define environment
# check the utils.py file for info on arguments

# Define network
encoder = BaselineEncoder(in_channels, feature_dim)
policy = BaselinePolicy(encoder, feature_dim, env.action_space.n)
policy = policy.cuda()
policy.load_state_dict(torch.load(total_path + '.pt'))

# image_split = ImageSplit(img_height, img_width, vert_splits, hor_splits)
# encoder = Encoder2(img_inp_layers,enc_out_dim_img)
# encoder_actions = ActionEncoder(act_in_features, act_l1_features, act_l2_features, act_out_features)
# pos_encoder_img = PositionalEncoder(enc_out_dim_img,hor_splits*vert_splits)
# pos_encoder_seq = PositionalEncoder(act_out_features,n_action_back)
# transformer_attention_img = MultiHeadAttention(img_heads, enc_out_dim_img)
# transformer_block_img = TransformerBlock_woa(transformer_attention_img, enc_out_dim_img, transf_dropout1, forward_scaling_img)
# transformer_attention_seq = MultiHeadAttention(seq_heads, act_out_features)
# transformer_block_seq = TransformerBlock_woa(transformer_attention_seq, act_out_features, transf_dropout2, forward_scaling_seq).cuda()
# data_augmentation = DataAugmentation(brightness, p_bright, contrast, p_contr, saturation, p_satur, hue, p_hue, augment_prob)
# policy = Policy(image_split, encoder, encoder_actions, pos_encoder_img, pos_encoder_seq, transformer_block_img, transformer_block_seq, vert_splits*hor_splits*enc_out_dim_img, policy_lin, env.action_space.n)
# policy.cuda()

train_df = pd.read_csv(total_path + '_train.csv')
train_df = np.array(train_df)
step_ns = [int(i) for i in train_df[:,1]]
mean_rewards = [float(i.split("tensor(")[1].split(')')[0]) for i in train_df[:,2]]
time_elapsed = [float(i) for i in train_df[:,3]]

plt.figure()
plt.plot(step_ns,mean_rewards)
plt.xlabel("Number of steps")
plt.ylabel("Average Reward gained")
plt.title("CR " + dirname)
plt.grid(True)
plt.savefig(total_path + '.pdf')

import imageio

# Make evaluation environment
eval_env = make_env(n_envs=test_n_envs, 
                env_name=test_env_name,
                num_levels=test_num_levels,
                start_level= test_start_level,
                use_backgrounds=test_use_backgrounds,
                normalize_obs=test_normalize_obs,
                normalize_reward=test_normalize_reward,
                seed=test_seed
                )

obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
done_vec = torch.tensor([0 for i in range(test_n_envs)])

# test_prev_actions = torch.zeros([test_n_envs,n_action_back, env.action_space.n])
# test_prev_actions2 = torch.zeros([test_n_envs,n_action_back, env.action_space.n])

n_steps_levels = np.zeros([test_n_envs, int(n_eval_levels/test_n_envs)])
reward_levels = np.zeros([test_n_envs, int(n_eval_levels/test_n_envs)])

n_test_steps = 0
while min(done_vec) < int(n_eval_levels/test_n_envs):
  
  # Use policy
  action, log_prob, value = policy.act(obs.cuda()) #Add , test_prev_actions.cuda() if prev actions are used

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  total_reward.append(torch.Tensor(reward))      
  
  if done_vec[0] < int(n_eval_levels/(test_n_envs*2)):
      # Render environment and store
      frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
      frames.append(frame)
  
  done_idx = [i for i, e in enumerate(done) if e == 1]
  for i in done_idx:
      if int(reward[i]) == 0:
          reward[i] = reward_dieing
      if not done_vec[i] > int(n_eval_levels/test_n_envs)-1:    
          reward_levels[i,done_vec[i]] = reward[i]
          if done_vec[i] > 0:
              n_steps_levels[i,done_vec[i]] = n_test_steps - sum(n_steps_levels[i,:done_vec[i]])
          else:
              n_steps_levels[i,done_vec[i]] = n_test_steps
  
  done_vec = done_vec + done
  #Update prev_actions
  # test_prev_actions2[:,1:,:] = test_prev_actions[:,:n_action_back-1,:]
  # test_prev_actions2[:,0,:] = nn.functional.one_hot(action, num_classes=15)
  # test_prev_actions = test_prev_actions2
  # test_prev_actions[done,:,:] = 0
  n_test_steps += 1
  
  if n_test_steps % 1000 == 0:
      print(done_vec)

# Calculate average return
n_steps_levels = np.reshape(n_steps_levels,(1,n_steps_levels.shape[0]*n_steps_levels.shape[1]))
reward_levels = np.reshape(reward_levels,(1,reward_levels.shape[0]*reward_levels.shape[1]))

eval_df = pd.DataFrame({'Level':range(n_steps_levels.shape[1]),
                        'N Steps':n_steps_levels[0,:], 
                        'Reward':reward_levels[0,:], 
                        })

eval_df2 = pd.DataFrame({
                        'Levels Complete':[sum(reward_levels[0,:] == 10)],
                        'Levels Dieing':[sum(np.logical_and(reward_levels[0,:] != 10, n_steps_levels[0,:] < 999))],
                        'Levels Incomplete':[sum(np.logical_and(reward_levels[0,:] != 10, n_steps_levels[0,:] >= 999))],
                        'Avg Steps Used':[np.mean(n_steps_levels[0,:])],
                        'Avg Steps - Completed Lvls':[np.mean([n_steps_levels[0,i] for i in range(n_steps_levels[0,:].shape[0]) if (reward_levels[0,i] == 10)])],
                        'Avg Steps - Incompleted Lvls':[np.mean([n_steps_levels[0,i] for i in range(n_steps_levels[0,:].shape[0]) if (reward_levels[0,i] != 10)])],
                        })


eval_df.to_csv(total_path + '_test_lvls.csv')
eval_df2.to_csv(total_path + '_test_stats.csv')

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave(total_path + '.mp4', frames, fps=15) 