import os
import numpy as np
import pandas as pd
import time
import mobilenet
import random
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob
import  tensorflow as tf
#from model import vggvox_model
from wav_reader import get_fft_spectrum
import constants as c
from scipy import spatial
import model
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def build_buckets(max_sec, step_sec, frame_step):
	buckets = {}
	frames_per_sec = int(1/frame_step)
	end_frame = int(max_sec*frames_per_sec)
	step_frame = int(step_sec*frames_per_sec)
	for i in range(0, end_frame+1, step_frame):
		s = i
		s = np.floor((s-7+2)/2) + 1  # conv1
		s = np.floor((s-3)/2) + 1  # mpool1
		s = np.floor((s-5+2)/2) + 1  # conv2
		s = np.floor((s-3)/2) + 1  # mpool2
		s = np.floor((s-3+2)/1) + 1  # conv3
		s = np.floor((s-3+2)/1) + 1  # conv4
		s = np.floor((s-3+2)/1) + 1  # conv5
		s = np.floor((s-3)/2) + 1  # mpool5
		s = np.floor((s-1)/1) + 1  # fc6
		if s > 0:
			buckets[i] = int(s)
	return buckets



def get_embedding(model, wav_file, max_sec):
	buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
	signal = get_fft_spectrum(wav_file, buckets)
	#signal = np.dstack((signal, signal, signal))
	start_time = time.time()
	embedding = np.squeeze(model.predict(signal.reshape(1, *signal.shape,1)))
	print("--- %s seconds ---" % (time.time() - start_time))

	return embedding


# def get_embedding_batch(model, wav_files, max_sec):
# 	return [ get_embedding(model, wav_file, max_sec) for wav_file in wav_files ]


def get_embeddings_from_list_file(model, list_file, max_sec):
	buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
	result = pd.read_csv(list_file, delimiter=",")
	result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(x, buckets))

	result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1))))
	return result[['filename','speaker','embedding']]

def distance(list1, list2):
    """Distance between two vectors."""
    squares = [(p-q) ** 2 for p, q in zip(list1, list2)]
    return sum(squares) ** .5

def get_id_result():
	print("Loading model weights")
	#m = model.vggvox_train_model()
	m = mobilenet.MobileNet((512, 300, 1), 1251, alpha=0.75, include_top=True, weights=None)


	m.load_weights("mbn_model.h5")
	#m.load_weights('vgg_model.h5')
	m = tf.keras.Model(m.input, m.layers[-2].output)

	m.summary()
	print("Processing samples")

	# load data
	veri_label = np.load("veri_label.npy")
	veri_enroll = np.load("veri_enroll.npy")
	veri_test = np.load("veri_test.npy")
	c = 0.0
	for i in range(len(veri_label)):
		data0 = get_embedding(m, veri_enroll[i], 3)
		data1 = get_embedding(m, veri_enroll[i], 3)
		data2 = get_embedding(m, veri_enroll[i], 3)
		data = (data0+data1+data2)/3.0


		th = 0
		#start_time = time.time()
		data0 = get_embedding(m, veri_test[i], 3)
		#print("--- %s seconds ---" % (time.time() - start_time))
		data1 = get_embedding(m, veri_test[i], 3)
		data2 = get_embedding(m, veri_test[i], 3)
		datav = (data0+data1+data2)/3.0

		distances = spatial.distance.cosine(data, datav)

		#print(distances)
		if distances < 0.3575:
			lab = 1
		else:
			lab = 0

		if lab == veri_label[i]:
			#print("correctly verifiy No:")
			print(i)
			c = c+1
		else:
			#print("verifiy  error No:")
			print(i)
		print("correct rate")
		print(c / (i+1))

	print("correct rate")
	print(c/len(veri_label))


if __name__ == '__main__':
	get_id_result()
