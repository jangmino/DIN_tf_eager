from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pickle
import random
import numpy as np
import sys
import tensorflow as tf
import pandas as pd
#import model

tfe = tf.contrib.eager
tfe.enable_eager_execution()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

train_batch_size = 32
test_batch_size = 512

# pkl_path = r'c:\Users\wmp\TensorFlow\DeepInterestNetwork\din\dataset.pkl'
#
# with open(pkl_path, 'rb') as f:
#   train_set = pickle.load(f)
#   test_set = pickle.load(f)
#   cate_list = pickle.load(f)
#   user_count, item_count, cate_count = pickle.load(f)

def device():
  return "/device:GPU:0" if tfe.num_gpus() else "/device:CPU:0"


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

LABEL_DIMENSION = 5

def random_dataset():
  batch_size = 64
  time_steps = 10
  alphabet = 50
  chars = tf.one_hot(
      tf.random_uniform(
          [batch_size, time_steps], minval=0, maxval=alphabet, dtype=tf.int32),
      alphabet)
  sequence_length = tf.constant(
      [time_steps for _ in range(batch_size)], dtype=tf.int64)
  labels = tf.random_normal([batch_size, LABEL_DIMENSION])
  return tf.data.Dataset.from_tensors((labels, chars, sequence_length))


class ModelDINTest(tf.test.TestCase):

  def testDataSet(self):
    # def gen():
    #   for u, ts, ij in test_set:
    #     yield (u, ts, ij[0], ij[1], len(ts))
    #
    # ds = tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64, tf.int64, tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])) )
    # ds = ds.padded_batch(16, padded_shapes=([], [None], [], [], []) )
    #
    # for (a,b,c,d,e) in tfe.Iterator(ds):
    #   print(a)
    #   print(b)
    #   print(c)

    d = {'r': [1, 1, 2], 'hist': [[1, 2, 3], [4, 5], [6]], 'pos': [7, 8, 9]}
    df = pd.DataFrame.from_dict(d)
    y = tf.data.Dataset.from_tensor_slices(dict(df))
    pass



if __name__ == "__main__":
  tf.test.main()
