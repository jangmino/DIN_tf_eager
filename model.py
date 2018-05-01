from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import pickle

import tensorflow as tf
tfe = tf.contrib.eager

#pkl_path = r'c:\Users\wmp\TensorFlow\DeepInterestNetwork\din\dataset.pkl'
pkl_path = r'/Users/jangmino/tensorflow/DeepInterestNetwork/din/dataset.pkl'

with open(pkl_path, 'rb') as f:
  train_set = pickle.load(f)
  test_set = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count = pickle.load(f)

train_batch_size = 32
test_batch_size = 512

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
    super(ItemBias, self).__init__(**kwargs)
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
    super(Attention, self).__init__()

    self.hidden_units = hidden_units

    self.attention_fc1 = layers.Dense(16, activation=tf.sigmoid)
    self.attention_fc2 = layers.Dense(8, activation=tf.sigmoid)
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
    super(ModelDIN, self).__init__()

    self.hidden_units = hidden_units
    embedding_dim = hidden_units // 2 # Warning: changed for speed
    self.item_embedding = Embedding(item_count, embedding_dim)
    self.item_bias = ItemBias(item_count)
    self.category_embedding = Embedding(len(category_list), embedding_dim)
    self.category_list = tf.convert_to_tensor(category_list, dtype=tf.int64)
    self.attention = Attention(hidden_units)

    self.bn_din = layers.BatchNormalization(name='bn_din')

    self.fc1 = layers.Dense(16)
    self.fc2 = layers.Dense(8)
    self.fc3 = layers.Dense(1)


  def call(self, inputs, training=False):
    """

    :param inputs: tuple(u, i, hist_i, sl) for training
    :param training:
    :return:
    """

    (u, i, hist_i, sl) = inputs

    ic = tf.gather(self.category_list, i)
    i_emb = tf.concat([self.item_embedding(i), self.category_embedding(ic)], axis=1)
    h_emb = tf.concat([self.item_embedding(hist_i), self.category_embedding(hist_i)], axis=2)

    hist = self.attention((i_emb, h_emb, sl), training=training)
    din_i = tf.concat([hist, i_emb], axis=-1)
    din_i = self.bn_din(din_i, training=training)

    outputs = self.fc1(din_i)
    outputs = self.fc2(outputs)
    outputs = self.fc3(outputs)
    predictions = self.item_bias(i) + outputs

    return predictions


def eval(model):

  def eval_gen():
    for u, ts, ij in test_set:
      yield (u, ts, ij[0], ij[1], len(ts))

  ds_eval = tf.data.Dataset.from_generator(eval_gen, (tf.int32, tf.int32, tf.int32, tf.int32, tf.int64), (tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])) )
  # ds_eval = ds_eval.padded_batch(test_batch_size, padded_shapes=([], [None], [], [], []) )
  ds_eval = ds_eval.batch(test_batch_size)

  auc_sum = 0.0
  test_size = 0

  total_time = 0
  start = time.time()
  for (u, ts, i, j, sl) in tfe.Iterator(ds_eval):
    # pred_i = model((u, i, ts, sl), training=False)
    # pred_j = model((u, j, ts, sl), training=False)

    test_size += 1 #int(u.shape[0])
    # auc_sum += float(tf.reduce_mean(tf.to_float(pred_i - pred_j > 0 )) * int(u.shape[0]))

  test_gauc = auc_sum / test_size
  total_time += (time.time() - start)
  sys.stderr.write("Elapsed {}\n".format(total_time))
  sys.stderr.flush()

  return test_gauc

def clip_gradients(grads_and_vars, clip_ratio):
  gradients, variables = zip(*grads_and_vars)
  clipped, _ = tf.clip_by_global_norm(gradients, clip_ratio)
  return zip(clipped, variables)

def train_one_epoch(model, optimizer, train_data):
  tf.train.get_or_create_global_step()

  def loss(inputs, labels):
    return tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=model(inputs, training=True),
        labels=labels)
    )

  val_grad_fn = tfe.implicit_value_and_gradients(loss)
  for (u, ts, i, y, sl) in tfe.Iterator(train_data):
    value, grads_and_vars = val_grad_fn((u, i, ts, sl), y)
    optimizer.apply_gradients(clip_gradients(grads_and_vars, 5), global_step=tf.train.get_global_step())

    if tf.grain.get_global_step() % 1000 == 0:
      eval(model)

def main(_):

  def train_gen():
    for u, ts, i, y in train_set:
      yield (u, ts, i, y, len(ts))

  ds_train = tf.data.Dataset.from_generator(train_gen,
                                      (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
                                      (tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])))
  ds_train = ds_train.padded_batch(train_batch_size, padded_shapes=([], [None], [], [], []))

  optimizer = tf.train.AdamOptimizer()
  m = ModelDIN(64, user_count, item_count, cate_list, use_dice=False)

  eval(m)
  for _ in range(50):
    train_one_epoch(m, optimizer, ds_train)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  FLAGS, unparsed = parser.parse_known_args()
  tfe.run(main=main, argv=[sys.argv[0]] + unparsed)
