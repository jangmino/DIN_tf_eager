from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import pickle
import random
import numpy as np
import sys
import glob
import pandas as pd

wepick_data_header = [
"v", "u", "seq", "rgtme", "dt", "label", "av", "bq", "dn", "dot", "dv", "dvcid", "g", "lid0",
"lid1", "lid2", "s", "ci", "dgid", "ef", "ls", "pe", "po", "pot", "ps", "set", "sst", "st",
"ti1", "ti2", "ti3", "ti4", "ti5", "tn1", "tn2", "tn3", "tn4", "tn5"
]

if __name__ == "__main__":
  data_dir = r'/Users/jangmino/tensorflow/DIN_tf_eager'
  dict = {}
  for fname in glob.glob(data_dir + '/*.csv'):
    df = pd.read_csv(fname, header=None, names=wepick_data_header)
    dict[fname] = df

  whole_df = pd.concat(df.values())

  print('123')