import os
import numpy as np
import  tensorflow as tf
import dg
from wav_reader import get_fft_spectrum
import constants as c
import json
import model
import mobilenet
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

# This function build the train data
def build_train_data():
    subjects = os.listdir('F:/wav/')
    subjects.sort()
    data = []
    label = {}
    test_data = []
    test_label = {}
    c = 0
    s = 0
    # -------------------for each person--------------------
    for i in subjects:
        #-----------------------for different speech----------------
        if i[0] != '.':
            speech = os.listdir('F:/wav/' + i)
            for j in speech:
                # -----------------------for different section------------------
                if j[0] != '.':
                    section = os.listdir('F:/wav/' + i + '/' +j)
                    for k in section:
                        if k[0] != '.' and k[-1] != 'y':
                            #set the data and label
                            name = 'F:/wav/' + i + '/' +j+'/'+k
                            if s <= 9:
                                #print(name)

                                f = get_embedding(name,3)
                                f = np.float32(f)
                                np.save(name, f)
                                data.append(name)
                                label[name] = int(c)

                                s = s+1
                                print(c)
                                print(i+j+k)
                            else:
                                f = get_embedding(name, 3)
                                f = np.float32(f)
                                np.save(name,f)
                                test_data.append(name)
                                test_label[name] = int(c)

                                s = 0
                                print(c)
                                print("test: " + i + j + k)

            c = c+1

    with open('label.json', 'w') as f:
        json.dump(label, f)

    with open('test_label.json', 'w') as f:
        json.dump(test_label, f)

    np.save("data", data)
    np.save("test_data",test_data)




#build_train_data()

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

#------------------------model--------------------------------
model = mobilenet.MobileNet((512, 300, 1), 1251, alpha=0.75, include_top=True, weights=None)
#---------------------------------------------------------------
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit_generator(generator=training_generator,
                    epochs=30,
                    validation_data=validation_generator,
                    validation_freq=2)


model.save_weights("mbn_model.h5")
print(history.history.keys())

np.save('acc.npy', history.history['acc'])
np.save('val_acc.npy', history.history['val_acc'])
np.save('loss.npy', history.history['loss'])
np.save('val_loss.npy', history.history['val_loss'])


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("mbn_training_acc_result_max.png")
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("mbn_training_loss_result_max.png")
plt.show()
