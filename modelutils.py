# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:55:32 2020

@author: Markus
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms 
import random
import numpy as np
import math
from projectutils import make_env, Storage, orthogonal_init
import matplotlib.image as mpimg

################ Print images code
# for i in storage.prev_obs[0,1]:
#     i = torch.transpose(i,0,2)
#     i = torch.transpose(i,0,1)
#     plt.imshow(i)
#     plt.show()


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

class Encoder2(nn.Module):
  def __init__(self, in_channels, encoder_out_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=576, out_features=encoder_out_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)

class ActionEncoder(nn.Module):
    def __init__(self, in_features, l1_features, l2_features, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, l1_features), nn.ReLU(),
            nn.Linear(l1_features, l2_features), nn.ReLU(),
            nn.Linear(l2_features, out_features), nn.ReLU(),
            )
        
        self.apply(orthogonal_init)
        
    def forward(self, x):
        return self.layers(x)
    
class BaselineEncoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)

class BaselinePolicy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value
    
class DataAugmentation(nn.Module):
    def __init__(self, brightness, p_bright, contrast, p_contr, saturation, p_satur, hue, p_hue, augment_prob):
        super().__init__()
        self.p_bright = p_bright
        self.p_contr = p_contr
        self.p_satur = p_satur
        self.p_hue = p_hue
        self.augment_prob = augment_prob
        self.to_tensor = transforms.ToTensor()
        self.to_pilimg = transforms.ToPILImage()
        self.brightness = transforms.Compose([transforms.ColorJitter(brightness = brightness)])
        self.contrast = transforms.ColorJitter(contrast = contrast)
        self.saturation = transforms.ColorJitter(saturation = saturation)
        self.hue = transforms.ColorJitter(hue = hue)
        
    def forward(self, x):
        img_list = [i for i in x]
        x = []
        for i in img_list:
            if random.random() < self.augment_prob:
                i = self.to_pilimg(i)
                if random.random() < self.p_bright:
                    i = self.brightness(i)
                if random.random() < self.p_contr:
                    i = self.contrast(i)
                if random.random() < self.p_satur:
                    i = self.saturation(i)
                if random.random() < self.p_hue:
                    i = self.hue(i)
                i = self.to_tensor(i)
                x.append(i)
            else:
                x.append(i)
            
        x = torch.stack(x,0)
        return x
        
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
        x = x * math.pow(self.d_model,1/3)
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

    def forward(self, values, keys, query, mask = None):
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
        
        if mask != None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
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
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, forward_scale * d_model),
            nn.ReLU(),
            nn.Linear(forward_scale * d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask = None):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class TransformerBlock_woa(nn.Module):
    def __init__(self, attention, d_model, dropout, forward_scale):
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, forward_scale * d_model),
            nn.ReLU(),
            nn.Linear(forward_scale * d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask = None):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Policy(nn.Module):
  def __init__(self, image_split, encoder, action_encoder, pos_encoder_img, pos_encoder_seq, transformer_block_img, transformer_block_seq, encoder_out_dim_img, encoder_out_dim_seq, num_actions):
    super().__init__()
    self.image_split = image_split
    self.encoder = encoder
    self.action_encoder = action_encoder
    self.pos_encoder_img = pos_encoder_img
    self.pos_encoder_seq = pos_encoder_seq
    self.transformer_block_img = transformer_block_img
    self.transformer_block_seq = transformer_block_seq
    self.linear = orthogonal_init(nn.Linear(encoder_out_dim_seq, int(encoder_out_dim_seq/2)), gain=.01)
    self.policy = orthogonal_init(nn.Linear(int(encoder_out_dim_seq/2), num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(int(encoder_out_dim_seq/2), 1), gain=1.)

  def act(self, x, actions):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x, actions)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x, actions):
    x = self.image_split(x)
    
    n = x.shape[0]
    splits = x.shape[1]
    
    x = torch.reshape(x,(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]))
    x = self.encoder(x)
    x = torch.reshape(x,(n,splits,x.shape[1]))
    x = self.pos_encoder_img(x)
    
    x = self.transformer_block_img(x,x,x)
    
    x = x.view(x.size(0), -1)
    
    n_act = actions.shape[0]
    act_back = actions.shape[1]
    
    act = torch.reshape(actions, (actions.shape[0]*actions.shape[1],actions.shape[2]))
    act = self.action_encoder(act)
    act = torch.reshape(act,(n_act,act_back,act.shape[1]))
    act = self.pos_encoder_seq(act)
    
    act = self.transformer_block_seq(act,act,act)
    
    act = act.view(act.size(0), -1)
    
    x = torch.cat([x,act],dim=1)
    
    x = F.relu(self.linear(x))
    
    logits = F.softmax(self.policy(x),dim=1)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value
