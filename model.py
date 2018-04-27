from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf

import tensorflow.contrib.eager as tfe


layers = tf.keras.layers

class Embedding(layers.Layer):
  """
  An Embedding layer.
  """
  def __init__(self, vocab_size, embedding_dim, **kwargs):
    super(Embedding, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim

  def build(self, _):
    self.embedding = self.add_variable(
        "embedding_kernel",
        shape=[self.vocab_size, self.embedding_dim],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        trainable=True)

  def call(self, x):
    return tf.nn.embedding_lookup(self.embedding, x)

class ItemBias(layers.Layer):
  def __init__(self, vocab_size, **kwargs):
    super(Embedding, self).__init__(**kwargs)
    self.vocab_size = vocab_size

  def build(self, _):
    self.item_b = self.add_variable(
        "item_bias",
        shape=[self.vocab_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=True)

  def call(self, x):
    return tf.gather(self.item_b, x)

class Attention(tf.keras.Model):
  def __init__(self, hidden_units):
    super(Attention, self).__init()

    self.hidden_units = hidden_units

    self.attention_fc1 = layers.Dense(80, activation=tf.sigmoid)
    self.attention_fc2 = layers.Dense(40, activation=tf.sigmoid)
    self.attention_fc3 = layers.Dense(1)
    self.attention_fc4 = layers.Dense(self.hidden_units)

  def call(self, inputs, training=False):
    (queries, keys, keys_length) = inputs
    queries_hidden_units = queries.get_shape().as_list()[-1]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
    din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)

    x = self.attention_fc1(din_all)
    x = self.attention_fc2(x)
    x = self.attention_fc3(x)

    outputs = tf.reshape(x, [-1, 1, tf.shape(keys)[1]])

    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
    key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]

    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]
    outputs = tf.reshape(outputs, [-1, self.hidden_units])

    outputs = self.attention_fc4(outputs)


    return outputs


class ModelDIN(tf.keras.Model):
  """
  An Eager Implementation of Deep Interest Network
  """

  def __init__(self, hidden_units, user_count, item_count, category_list, use_dice=False):
    super(ModelDIN, self).__init()

    self.hidden_units = hidden_units
    embedding_dim = hidden_units // 2
    self.item_embedding = Embedding(item_count, embedding_dim)
    self.item_bias = ItemBias(item_count)
    self.category_embedding = Embedding(len(category_list), embedding_dim)
    self.category_list = tf.convert_to_tensor(category_list, dtype=tf.int64)
    self.attention = Attention(hidden_units)

    self.bn_din = layers.BatchNormalization(name='bn_din')

    self.fc1 = layers.Dense(80)
    self.fc2 = layers.Dense(40)
    self.fc3 = layers.Dense(1)



  def call(self, inputs, training=False):
    """

    :param inputs: tuple(u, i, y, hist_i, sl) for training
    :param training:
    :return:
    """

    (u, i, y, hist_i, sl) = inputs

    ic = tf.gather(self.category_list, i)
    i_emb = tf.concat([self.item_embedding(i), self.category_embedding(ic)], axis=1)
    h_emb = tf.concat([self.item_embedding(hist_i), self.category_embedding(hist_i)])

    hist = self.attention((i_emb, h_emb, sl), training=training)
    din_i = tf.concat([hist, i_emb], axis=-1)
    din_i = self.bn_din(din_i, training=training)

    outputs = self.fc1(din_i)
    outputs = self.fc2(outputs)
    outputs = self.fc3(outputs)

    pass
