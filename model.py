import scipy.io as sio
import numpy as np
import tensorflow as tf

import constants as c

# Block of layers: Conv --> BatchNorm --> ReLU --> Pool
def conv_bn_pool(inp_tensor,layer_idx,conv_filters,conv_kernel_size,conv_strides,conv_pad,
	pool='',pool_size=(2, 2),pool_strides=None,
	conv_layer_prefix='conv'):
	x = tf.keras.layers.ZeroPadding2D(padding=conv_pad,name='pad{}'.format(layer_idx))(inp_tensor)
	x = tf.keras.layers.Conv2D(filters=conv_filters,kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix,layer_idx))(x)
	x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-5, name='bn{}'.format(layer_idx))(x)
	x = tf.keras.layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
	if pool == 'max':
		x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,strides=pool_strides,name='mpool{}'.format(layer_idx))(x)
	elif pool == 'avg':
		x = tf.keras.layers.AveragePooling2D(pool_size=pool_size,strides=pool_strides,name='apool{}'.format(layer_idx))(x)
	return x


# Block of layers: Conv --> BatchNorm --> ReLU --> Dynamic average pool (fc6 -> apool6 only)
def conv_bn_dynamic_apool(inp_tensor,layer_idx,conv_filters,conv_kernel_size,conv_strides,conv_pad,
	conv_layer_prefix='conv'):
	x = tf.keras.layers.ZeroPadding2D(padding=conv_pad,name='pad{}'.format(layer_idx))(inp_tensor)
	x = tf.keras.layers.Conv2D(filters=conv_filters,kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix,layer_idx))(x)
	x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-5, name='bn{}'.format(layer_idx))(x)
	x = tf.keras.layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
	x = tf.keras.layers.GlobalAveragePooling2D(name='gapool{}'.format(layer_idx))(x)
	x = tf.keras.layers.Reshape((1,1,conv_filters),name='reshape{}'.format(layer_idx))(x)
	return x

def vggvox_train_model():
	inp = tf.keras.layers.Input((512,300,1),name='input')
	x = conv_bn_pool(inp,layer_idx=1,conv_filters=96,conv_kernel_size=(7,7),conv_strides=(2,2),conv_pad=(1,1),
		pool='max',pool_size=(3,3),pool_strides=(2,2))
	x = conv_bn_pool(x,layer_idx=2,conv_filters=256,conv_kernel_size=(5,5),conv_strides=(2,2),conv_pad=(1,1),
		pool='max',pool_size=(3,3),pool_strides=(2,2))
	x = conv_bn_pool(x,layer_idx=3,conv_filters=384,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
	x = conv_bn_pool(x,layer_idx=4,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
	x = conv_bn_pool(x,layer_idx=5,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1),
		pool='max',pool_size=(5,3),pool_strides=(3,2))

	x = conv_bn_dynamic_apool(x,layer_idx=6,conv_filters=4096,conv_kernel_size=(9,1),conv_strides=(1,1),conv_pad=(0,0),
		conv_layer_prefix='fc')

	x = conv_bn_pool(x,layer_idx=7,conv_filters=1024,conv_kernel_size=(1,1),conv_strides=(1,1),conv_pad=(0,0),
		conv_layer_prefix='fc')

	x = tf.keras.layers.Lambda(lambda y: tf.keras.backend.l2_normalize(y, axis=3), name='norm')(x)
	x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=(1,1), padding='valid', name='fc8')(x)
	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.Dense(1251, activation='softmax')(x)
	m = tf.keras.Model(inp, x, name='VGGVox')
	return m

