import numpy as np 
import skimage

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.utils.generic_utils import get_custom_objects
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from keras.layers import Activation
import random

def add_noise(img,  mode = 'gaussian', var = 0.0001):
	m = np.mean(img)
	img = skimage.util.random_noise(img, mode=mode, seed=None, clip=True, mean = m, var = var)
	img[img<0] = 0
	return img

class sparsemax(Layer):

	def call(self, logits):
		logits = ops.convert_to_tensor(logits, name="logits")
		obs = array_ops.shape(logits)[0]
		dims = array_ops.shape(logits)[1]
		z = logits - math_ops.reduce_mean(logits, axis=1)[:, array_ops.newaxis]

		# sort z
		z_sorted, _ = nn.top_k(z, k=dims)

		# calculate k(z)
		z_cumsum = math_ops.cumsum(z_sorted, axis=1)
		k = math_ops.range(1, math_ops.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
		z_check = 1 + k * z_sorted > z_cumsum
		k_z = math_ops.reduce_sum(math_ops.cast(z_check, dtypes.int32), axis=1)

		# calculate tau(z)
		indices = array_ops.stack([math_ops.range(0, obs), k_z - 1], axis=1)
		tau_sum = array_ops.gather_nd(z_cumsum, indices)
		tau_z = (tau_sum - 1) / math_ops.cast(k_z, logits.dtype)

		# calculate p
		return math_ops.maximum(math_ops.cast(0, logits.dtype), z - tau_z[:, array_ops.newaxis])

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1])

	def get_config(self):
		return {"name":self.__class__.__name__}


def random_transpose(img):
	p = random.random()
	if p<0.5:
		img = np.transpose(img)
	return img
