import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob
import  tensorflow as tf
#from model import vggvox_model
from wav_reader import get_fft_spectrum
import constants as c
import mobilenet
import model
from scipy import spatial

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
	embedding = np.squeeze(model.predict(signal.reshape(1,*signal.shape,1)))
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
	# ---------------------------------------------------uncomment to switch model
	m = mobilenet.MobileNet((512, 300, 1), 1251, alpha=0.75, include_top=True, weights=None)
	m.load_weights('mbn_model.h5')
	# m = model.vggvox_train_model()
	# m.load_weights('vgg_model.h5')

	#---------------------------------------------------
	m = tf.keras.Model(m.input, m.layers[-2].output)

	m.summary()
	print("Processing samples")
	th = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	tp = [0,0,0,0,0,0,0,0,0,0]
	fp = [0,0,0,0,0,0,0,0,0,0]
	p = 0
	# load data
	veri_label = np.load("veri_label.npy")
	veri_enroll = np.load("veri_enroll.npy")
	veri_test = np.load("veri_test.npy")
	#len(veri_label)
	for i in range(len(veri_label)):
		data0 = get_embedding(m, veri_enroll[i], 3)
		data1 = get_embedding(m, veri_test[i], 3)


		for n in range(4):
			data0 = data0 + get_embedding(m, veri_enroll[i], 3)
			data1 = data1 + get_embedding(m, veri_test[i], 3)
			distances = spatial.distance.cosine(data0/5, data1/5)


		if veri_label[i] == 1:
			p = p+1
			print(distances)
		#else:
			#print(distances)
		for k in range(len(tp)):
			if veri_label[i] == 1 and distances < th[k]:
				tp[k] = tp[k]+1
			if veri_label[i] == 0 and distances < th[k]:
				fp[k] = fp[k]+1
	for j in range(len(tp)):
		print("Th = ", end = '')
		print(th[j])
		print("true positive: ", end = '')
		tpr =  "{:.2f}".format(tp[j]/p)
		print(tpr)
		print("false positive: ", end = '')
		fpr = "{:.2f}".format( fp[j] / (len(veri_label)-p))
		print(fpr)

if __name__ == '__main__':
	get_id_result()
