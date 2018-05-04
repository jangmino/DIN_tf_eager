from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import pickle
import numpy as np

import tensorflow as tf
tfe = tf.contrib.eager

pkl_path = r'c:\Users\wmp\TensorFlow\DeepInterestNetwork\din\dataset.pkl'
#pkl_path = r'/Users/jangmino/tensorflow/DeepInterestNetwork/din/dataset.pkl'

with open(pkl_path, 'rb') as f:
  train_set = pickle.load(f)
  test_set = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count = pickle.load(f)

train_batch_size = 512
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
    sys.stdout.flush()
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

  def __init__(self, hidden_units, user_count, item_count, category_list, device="/cpu:0", use_dice=False):
    super(ModelDIN, self).__init__()

    self.device=device
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
    (u, i, i_c, hist_i, hist_c, sl) = inputs

    #ic = tf.gather(self.category_list, i)
    i_emb = tf.concat([self.item_embedding(i), self.category_embedding(i_c)], axis=1)
    h_emb = tf.concat([self.item_embedding(hist_i), self.category_embedding(hist_c)], axis=2)

    hist = self.attention((i_emb, h_emb, sl), training=training)
    din_i = tf.concat([hist, i_emb], axis=-1)
    din_i = self.bn_din(din_i, training=training)

    outputs = self.fc1(din_i)
    outputs = self.fc2(outputs)
    outputs = self.fc3(outputs)
    outputs = tf.reshape(outputs, [-1])
    predictions = self.item_bias(i) + outputs

    return predictions

def parse_train(line):
  """
  reviewerID, hist, next, y
  :param line:
  :return:
  """
  items = tf.string_split([line], ",").values
  reviewerID = tf.string_to_number(items[0], out_type=tf.int64)
  hists_ = tf.string_split([items[1]], ":").values
  hist = tf.string_to_number(hists_, out_type=tf.int64)
  i = tf.string_to_number(items[2], out_type=tf.int64)
  y = tf.string_to_number(items[3], out_type=tf.float32)

  length = tf.cast(tf.shape(hist)[0], dtype=tf.int64)
  return reviewerID, hist, i, y, length

def parse_eval(line):
  """
  reviewerID, hist, i, j
  :param line:
  :return:
  """
  items = tf.string_split([line], ",").values
  reviewerID = tf.string_to_number(items[0], out_type=tf.int64)
  hists_ = tf.string_split([items[1]], ":").values
  hist = tf.string_to_number(hists_, out_type=tf.int64)
  i = tf.string_to_number(items[2], out_type=tf.int64)
  j = tf.string_to_number(items[3], out_type=tf.int64)

  length = tf.cast(tf.shape(hist)[0], dtype=tf.int64)
  return reviewerID, hist, i, j, length


def eval(model):

  # def eval_gen():
  #   for u, ts, ij in test_set:
  #     yield (u, ts, ij[0], ij[1], len(ts))
  #
  # ds_eval = tf.data.Dataset.from_generator(eval_gen, (tf.int32, tf.int32, tf.int32, tf.int32, tf.int64), (tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])) )
  # # ds_eval = ds_eval.padded_batch(test_batch_size, padded_shapes=([], [None], [], [], []) )
  # ds_eval = ds_eval.batch(test_batch_size)

  ds_eval = tf.data.TextLineDataset('din_test.csv').skip(1).map(parse_eval).padded_batch(
    test_batch_size, padded_shapes=([], [None], [], [], [])
    )

  auc_sum = 0.0
  test_size = 0

  total_time = 0
  total_model = 0
  start = time.time()
  for (u, ts, i, j, sl) in ds_eval:
    hist_c = tf.convert_to_tensor(list(map(lambda x: cate_list[x], ts.numpy())), dtype=tf.int64)
    i_c = tf.convert_to_tensor(list(map(lambda x: cate_list[x], i.numpy())), dtype=tf.int64)
    j_c = tf.convert_to_tensor(list(map(lambda x: cate_list[x], j.numpy())), dtype=tf.int64)

    inf_start = time.time()
    pred_i = model((u, i, i_c, ts, hist_c, sl), training=False)
    pred_j = model((u, j, j_c, ts, hist_c, sl), training=False)
    inf_end = time.time()
    total_model += inf_end - inf_start

    test_size += int(u.shape[0])
    auc_sum += float(tf.reduce_mean(tf.to_float(pred_i - pred_j > 0 )) * int(u.shape[0]))

  test_gauc = auc_sum / test_size
  total_time += (time.time() - start)
  sys.stderr.write("Elapsed total {} : model {}\n".format(total_time, total_model))
  sys.stderr.flush()

  return test_gauc

def clip_gradients(grads_and_vars, clip_ratio):
  gradients, variables = zip(*grads_and_vars)
  clipped, _ = tf.clip_by_global_norm(gradients, clip_ratio)
  return zip(clipped, variables)

def train_one_epoch(epoch_i, model, optimizer, train_data, step_counter, log_interval=10):

  def loss(inputs, labels):
    return tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=model(inputs, training=True),
        labels=labels)
    )

  val_grad_fn = tfe.implicit_value_and_gradients(loss)

  loss_sum = 0
  n_step = 0

  category_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

  for (u, ts, i, y, sl) in train_data:
    hist_c = tf.convert_to_tensor(list(map(lambda x: cate_list[x], ts.numpy())), dtype=tf.int64)
    i_c = tf.convert_to_tensor(list(map(lambda x: cate_list[x], i.numpy())), dtype=tf.int64)
    with tf.contrib.summary.record_summaries_every_n_global_steps(log_interval):
      with tf.device("/cpu:0"):
        value, grads_and_vars = val_grad_fn((u, i, i_c, ts, hist_c, sl), y)
      tf.contrib.summary.scalar("loss", value)
      optimizer.apply_gradients(clip_gradients(grads_and_vars, 5), global_step=step_counter)
      loss_sum += value

    n_step += 1

    if tf.train.get_global_step().numpy() % 100 == 0:
      test_gauc = eval(model)
      print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f' %
            (epoch_i, tf.train.get_global_step().numpy(),
             loss_sum / 100, test_gauc))
      sys.stdout.flush()
      loss_sum = 0
      n_step = 0


def main(_):

  # def train_gen():
  #   for u, ts, i, y in train_set:
  #     yield (u, ts, i, y, len(ts))
  #
  # ds_train = tf.data.Dataset.from_generator(train_gen,
  #                                     (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
  #                                     (tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])))
  # ds_train = ds_train.padded_batch(train_batch_size, padded_shapes=([], [None], [], [], []))

  if FLAGS.no_gpu or tfe.num_gpus() <= 0:
    print(tfe.num_gpus())
    device = "/cpu:0"
  else:
    device = "/gpu:0"
  print("Using device %s." % device)

  log_dir = os.path.join(FLAGS.dir, "summaries")
  tf.gfile.MakeDirs(log_dir)
  train_summary_writer = tf.contrib.summary.create_file_writer(
      os.path.join(log_dir, "train"), flush_millis=10000)
  # test_summary_writer = tf.contrib.summary.create_file_writer(
  #     os.path.join(log_dir, "eval"), flush_millis=10000, name="eval")

  ds_train = tf.data.TextLineDataset('din_train.csv').skip(1).map(parse_train).padded_batch(
    train_batch_size, padded_shapes=([], [None], [], [], [])
    )

  # tf.keras.backend.set_session(tf.Session(config=tf.ConfigProto(
  #   gpu_options=tf.GPUOptions(allow_growth=True),
  #   log_device_placement=True))
  # )


  model_objects = {
    'model': ModelDIN(64, user_count, item_count, cate_list, device=device, use_dice=False),
    'optimizer': tf.train.AdamOptimizer(FLAGS.learning_rate),
    'step_counter': tf.train.get_or_create_global_step(),
  }

  checkpoint_prefix = os.path.join(FLAGS.dir, 'ckpt')
  latest_cpkt = tf.train.latest_checkpoint(FLAGS.dir)
  if latest_cpkt:
    print('Using latest checkpoint at ' + latest_cpkt)
  checkpoint = tfe.Checkpoint(**model_objects)
  # Restore variables on creation if a checkpoint exists.
  checkpoint.restore(latest_cpkt)

  #with tf.device(device):
  test_gauc = eval(model_objects['model'])
  for i in range(FLAGS.num_epochs):
    start = time.time()
    with train_summary_writer.as_default():
      train_one_epoch(epoch_i=i, train_data=ds_train, log_interval=FLAGS.log_interval, **model_objects)
      end = time.time()
      checkpoint.save(checkpoint_prefix)
      print('\nTrain time for epoch #%d (step %d): %f' %
            (checkpoint.save_counter.numpy(),
             checkpoint.step_counter.numpy(),
             end - start))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--dir",
      type=str,
      default="./dnn_eager/",
      help="Directory to locate data files and save logs.")
  parser.add_argument(
      "--log_interval",
      type=int,
      default=10,
      metavar="N",
      help="Log training loss every log_interval batches.")
  parser.add_argument(
      "--num_epochs", type=int, default=20, help="Number of epochs to train.")
  parser.add_argument(
      "--rnn_cell_sizes",
      type=int,
      nargs="+",
      default=[256, 128],
      help="List of sizes for each layer of the RNN.")
  parser.add_argument(
      "--batch_size",
      type=int,
      default=64,
      help="Batch size for training and eval.")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.001,
      help="Learning rate to be used during training.")
  parser.add_argument(
      "--no_gpu",
      action="store_true",
      default=False,
      help="Disables GPU usage even if a GPU is available.")

  FLAGS, unparsed = parser.parse_known_args()
  tfe.enable_eager_execution(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)),
                                                   device_policy=tfe.DEVICE_PLACEMENT_SILENT)
  # tfe.enable_eager_execution()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
