import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, datasets, models, Sequential
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Network(Model):

  def __init__(self, out_dim=16):
    super(Network, self).__init__(name='Network')
    self.out_dim = out_dim
    # Define your layers here.
    self.conv1 = layers.Conv2D(filters=16, kernel_size=(8,8), activation="relu")
    self.pool1 = layers.MaxPool2D((2, 2))
    self.conv2 = layers.Conv2D(filters=7, kernel_size=(5,5), activation="relu")
    self.pool2 = layers.MaxPool2D((2, 2))
    self.flat = layers.Flatten()
    self.fc1 = layers.Dense(256, activation=None)
    self.fc2 = layers.Dense(out_dim, activation=None)

  def call(self, inputs):
    # Define your forward pass here.
    x = self.conv1(inputs)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.flat(x)
    x = self.fc1(x)
    x = self.fc2(x)
    return x

def triplet_pair_loss(triplet_batch):
  batch_size = triplet_batch.shape[0]
  diff_pos = triplet_batch[0:batch_size:3] - triplet_batch[1:batch_size:3]
  diff_neg = triplet_batch[0:batch_size:3] - triplet_batch[2:batch_size:3]
  dist_pos = tf.nn.l2_loss(diff_pos) * 2
  dist_neg = tf.nn.l2_loss(diff_neg) * 2

  m = 0.01
  ratio = dist_neg / (dist_pos + m)
  triplet_loss = tf.reduce_sum(tf.maximum(0.0, (1.0-ratio)))
  pair_loss = tf.reduce_sum(dist_pos)

  return triplet_loss + pair_loss

#def test_function(test_dataset, database, model):
    

@tf.function
def train(model, dataset, optimizer):
  batch_count = dataset.shape[0]
  loss_history = []
  histograms = []
  for i in range(batch_count):
    with tf.GradientTape() as tape:
      preds = model(dataset[i], training=True)
      loss = triplet_pair_loss(preds)
      # loss_history.append(loss)
      tf.print("Loss:", loss, output_stream = sys.stdout)
      #print("[Iteration: {}, Loss: {}]".format(i, loss))
    if (i % 1000) == 0:
      pass
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def normalize(tensor):
  """Perform zero mean - unit variance normalization of each channel of input of the form: HxWxC."""
  mean = np.mean(tensor, axis=(0,1))
  std = np.std(tensor, axis=(0,1))
  return (tensor - mean) / std

# def confusion_matrix():
#TODO:
#   flights = sns.load_dataset("flights") # Pandas instance
#   flights = flights.pivot("month", "year", "passengers") # Row, col, val
#   ax = sns.heatmap(flights, annot=True, annot_kws={"fontsize":5}, fmt="d") # Annotate cells with ints
#   print(ax)
#   plt.show()

# def match():
#TODO:
#   bf = cv2.BFMatcher()
#   matches = bf.knnMatch(des1, des2, k=2)