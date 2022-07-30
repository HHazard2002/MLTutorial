import os
from attr import validate
import numpy as np
import matplotlib.pylot as plt
import tensorflow as tf
keras = tf.keras

import tensorflow_dataset as tfds
tfds.disable_progress_bar()