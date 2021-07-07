import os
import numpy as np
import  tensorflow as tf
import dg
from wav_reader import get_fft_spectrum
import constants as c
import json
import model
import matplotlib.pyplot as plt

def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1 / frame_step)
    end_frame = int(max_sec * frames_per_sec)
    step_frame = int(step_sec * frames_per_sec)
    for i in range(0, end_frame + 1, step_frame):
        s = i
        s = np.floor((s - 7 + 2) / 2) + 1  # conv1
        s = np.floor((s - 3) / 2) + 1  # mpool1
        s = np.floor((s - 5 + 2) / 2) + 1  # conv2
        s = np.floor((s - 3) / 2) + 1  # mpool2
        s = np.floor((s - 3 + 2) / 1) + 1  # conv3
        s = np.floor((s - 3 + 2) / 1) + 1  # conv4
        s = np.floor((s - 3 + 2) / 1) + 1  # conv5
        s = np.floor((s - 3) / 2) + 1  # mpool5
        s = np.floor((s - 1) / 1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets


def get_embedding(wav_file, max_sec):
    buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
    signal = get_fft_spectrum(wav_file, buckets)
    embedding = signal.reshape(*signal.shape, 1)
    return embedding


data = np.load("data.npy")
test_data = np.load("test_data.npy")
label = {}
test_label = {}

with open('label.json') as f:
    label = json.load(f)

with open('test_label.json') as f:
    test_label = json.load(f)


params = {'dim': (512,300),
          'batch_size': 32,
          'n_classes': 1251,
          'n_channels': 1,
          'shuffle': True}



# Generators
training_generator = dg.DataGenerator(data, label, **params)
validation_generator = dg.DataGenerator(test_data, test_label, **params)

model = model.vggvox_train_model()
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit_generator(generator=training_generator,
                    epochs=12,
                    validation_data=validation_generator,
                    validation_freq=1)


model.save_weights("vgg_model.h5")
print(history.history.keys())
np.save('vacc.npy', history.history['acc'])
np.save('vval_acc.npy', history.history['val_acc'])
np.save('vloss.npy', history.history['loss'])
np.save('vval_loss.npy', history.history['val_loss'])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("vgg_m_training_acc_result.png")
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("vgg_m_training_loss_result.png")
plt.show()
