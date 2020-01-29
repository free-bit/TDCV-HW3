import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletNet(nn.Module):

  def __init__(self, out_dim=16):
    super(TripletNet, self).__init__()
    # Layers
    self.layers = nn.Sequential(
      # NOTE: Input format: NxCxHxW
      nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(8,8)),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2),
      nn.Conv2d(in_channels=16, out_channels=7, kernel_size=(5,5)),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 2), stride=2),
      nn.Flatten(),
      nn.Linear(in_features=1008, out_features=256),
      nn.Linear(in_features=256, out_features=out_dim)
    )

  def forward(self, x):
    # Forward pass
    x = self.layers(x)
    return x

def triplet_pair_loss(triplet_batch):
  '''
  Calculate triplet and pair losses based on given formula.
  Return the joint loss obtained by summing triplet and pair losses.
  '''
  batch_size = triplet_batch.shape[0]      # BS: batch_size
  diff_pos = triplet_batch[0:batch_size:3] - triplet_batch[1:batch_size:3] # BSx16 -> BSx16, anchors - pullers
  diff_neg = triplet_batch[0:batch_size:3] - triplet_batch[2:batch_size:3] # BSx16 -> BSx16, anchors - pushers
  dist_pos = torch.sum(diff_pos**2, dim=1) # BSx16 -> BSx1, take the square of L2 norm of each vector separately
  dist_neg = torch.sum(diff_neg**2, dim=1) # BSx16 -> BSx1, take the square of L2 norm of each vector separately

  m = 0.01
  ratio = dist_neg / (dist_pos + m)            # BSx1 -> BSx1
  max_values = torch.clamp((1.0-ratio), min=0) # BSx1 -> BSx1, clamp used for max(0, 1-ratio)
  triplet_loss = torch.sum(max_values, dim=0)  # BSx1 -> 1x1, sum the max values (one max value per vector)
  pair_loss = torch.sum(dist_pos, dim=0)       # BSx1 -> 1x1, sum the square of L2 norm of every vector

  return triplet_loss + pair_loss

def normalize(tensor):
  """Perform zero mean - unit variance normalization of each channel of input of the form: HxWxC."""
  mean = np.mean(tensor, axis=(0,1))
  std = np.std(tensor, axis=(0,1))
  return (tensor - mean) / std