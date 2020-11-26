# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:28:39 2020

@author: Markus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from projectutils import make_env, Storage, orthogonal_init

total_steps = 1e6
num_envs = 32
num_levels = 10
num_steps = 256
num_epochs = 3
batch_size = 256
eps = .2
grad_eps = .5
value_coef = .3
entropy_coef = .1

env = make_env(n_envs=32, 
               env_name='starpilot',
               num_levels=10,
               start_level= 0,
               use_backgrounds=False,
               normalize_obs=False,
               normalize_reward=True,
               seed=0
               )

print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ImageSplit(nn.Module):
    def __init__(self,img_height,img_width,vertical_splits, horizontal_splits):    
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.vertical_splits = vertical_splits
        self.horizontal_splits = horizontal_splits
        
        assert( float(img_height // vertical_splits) == (img_height / vertical_splits)), "img_heaight should be divisible by vertical_splits"
        assert( float(img_width // horizontal_splits) == (img_width / horizontal_splits)), "img_width should be divisible by horizontal_splits"
        
        self.split_height = int(img_height / vertical_splits)
        self.split_length = int(img_width / horizontal_splits)
        self.image_split = []
    
    def forward(self,x):
        self.image_split = []
            
        for i in range(self.vertical_splits):
            for j in range(self.horizontal_splits):
                self.image_split.append(x[:,:,i*self.split_length:(i+1)*self.split_length,j*self.split_height:(j+1)*self.split_height])
        
        return torch.stack(self.image_split,1)

def CalculateConvDim(dimension, kernel_size, stride, padding, pool_stride, pool_kernel, pool_padding):
    if (dimension - kernel_size) % stride != 0:
        print("Kernel_size, Stride and image dimension does not fit.")
        return False
    else:
        AfterConv = 1 + int((dimension-kernel_size+padding*2)/stride)

    if (AfterConv - pool_kernel) % pool_stride != 0:
        print("Pool Stride and Kernel does not fit AfterConv dimension")
        return False
    
    AfterPool = 1 + int((AfterConv-pool_kernel+pool_padding*2)/pool_stride)
    return AfterPool
        

class Encoder(nn.Module):
  def __init__(self, in_channels, encoder_out_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=encoder_out_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)

class Encoder2(nn.Module):
  def __init__(self, in_channels, encoder_out_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=576, out_features=encoder_out_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_splits = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_splits, d_model)
        for pos in range(max_splits):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.pow(self.d_model,1/2)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len].clone().detach().cuda()
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.head_dim = d_model // heads
        self.heads = heads
        
        assert (self.head_dim * self.heads == self.d_model), "dimension of embeddings, should be divisible by heads"
        
        self.q_linear = nn.Linear(self.head_dim, self.head_dim)
        self.v_linear = nn.Linear(self.head_dim, self.head_dim)
        self.k_linear = nn.Linear(self.head_dim, self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(self.heads*self.head_dim, self.d_model)

    def forward(self, values, keys, query):
        # Get number of env running at the same time
        batch_n = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Reshape into head dimensions
        values = values.reshape(batch_n, value_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_n, key_len, self.heads, self.head_dim)
        query = query.reshape(batch_n, query_len, self.heads, self.head_dim)

        values = self.v_linear(values)  # (batch_n, value_len, heads, head_dim)
        keys = self.k_linear(keys)  # (batch_n, key_len, heads, head_dim)
        queries = self.q_linear(query)  # (batch_n, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # Equivalent til prikke alle 64,16 ved q med 64,16 ved k, og få 16x16 matricer ud

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.d_model ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            batch_n, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out
        
class TransformerBlock(nn.Module):
    def __init__(self, attention, d_model, dropout, forward_scale):
        super(TransformerBlock, self).__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, forward_scale * d_model),
            nn.ReLU(),
            nn.Linear(forward_scale * d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Policy(nn.Module):
  def __init__(self, image_split, encoder, pos_encoder, transformer_block, encoder_out_dim, num_actions):
    super().__init__()
    self.image_split = image_split
    self.encoder = encoder
    self.pos_encoder = pos_encoder
    self.transformer_block = transformer_block
    self.policy = orthogonal_init(nn.Linear(encoder_out_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(encoder_out_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    x = self.image_split(x)
    
    n = x.shape[0]
    splits = x.shape[1]
    
    x = torch.reshape(x,(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
    x = self.encoder(x)
    x = torch.reshape(x,(n,splits,x.shape[1]))
    x = self.pos_encoder(x)
    
    x = self.transformer_block(x,x,x)
    
    x = x.view(x.size(0), -1)
    
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value


# Define environment
# check the utils.py file for info on arguments

# Define network
encoder_out_dim = 128
split_n = 4
image_split = ImageSplit(64, 64, split_n, split_n)
encoder = Encoder2(3,encoder_out_dim)
pos_encoder = PositionalEncoder(128,16)
transformer_attention = MultiHeadAttention(4, encoder_out_dim)
transformer_block = TransformerBlock(transformer_attention, encoder_out_dim, 0.1, 4)
policy = Policy(image_split, encoder, pos_encoder, transformer_block, encoder_out_dim*split_n*split_n, env.action_space.n)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)




# Run training
obs = env.reset()
step_ns = []
mean_rewards = []
step = 0
while step < total_steps:

  # Use policy to collect data for "num_steps" steps in the environment. 
  policy.eval()
  for _ in range(num_steps):
    # Use policy
    action, log_prob, value = policy.act(obs)
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    
    # Update current observation
    obs = next_obs


  # Add the last observation to collected data
  _, _, value = policy.act(obs)
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # Optimize policy
  policy.train()
  
  for epoch in range(num_epochs):

    # Iterate over batches of transitions
    generator = storage.get_generator(batch_size)
    for batch in generator:
      b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

      # Get current policy outputs
      new_dist, new_value = policy(b_obs)
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
  step += num_envs * num_steps
  print(f'Step: {step}\tMean reward: {storage.get_reward()}')
  step_ns.append(step)
  mean_rewards.append(storage.get_reward())

print('Completed training!')
torch.save(policy.state_dict, 'D:/OneDrive - Danmarks Tekniske Universitet/Studie/5. Semester/Deep Learning/Projekt/Model1_states/checkpoint_3600kM1.pt')

import imageio

# Make evaluation environment
eval_env = make_env(num_envs, start_level=num_levels, num_levels=num_levels)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
for _ in range(1024):

  # Use policy
  action, log_prob, value = policy.act(obs)

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  total_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('D:/OneDrive - Danmarks Tekniske Universitet/Studie/5. Semester/Deep Learning/Projekt/Model1_states/vid3600kM1.mp4', frames, fps=25)