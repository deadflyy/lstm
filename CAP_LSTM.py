import numpy as np
import tensorflow as tf
import time
import os
import random
from collections import namedtuple
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense
from collections import Counter
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
flags = tf.app.flags
flags.DEFINE_string("mode", "train", "run mode [train]")
flags.DEFINE_integer("epoch", 10, "Epoch to train [10]")
flags.DEFINE_integer("num_class", 2, "class to train [2]")
flags.DEFINE_integer("num_steps", 128, "num steps [100]")
flags.DEFINE_integer("lstm_size", 128, "hidden size of lstm [100]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [64]")
flags.DEFINE_integer("num_layers", 10, "hidden layers of lstm. [2]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.001]")
flags.DEFINE_float("keep_prob", 0.5, "Learning rate of for adam [0.001]")
flags.DEFINE_string("dataset", "train20171225.txt",
                    "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples",
                    "Directory name to save the image samples [samples]")

FLAGS = flags.FLAGS




def train(points, seqlength, targets,checkpointer):
    save_every_n = 100
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(FLAGS.num_steps, 3)))

    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(LSTM(64, return_sequences=True))
    # model.add(LSTM(64, return_sequences=True, stateful=True,
    #            batch_input_shape=(FLAGS.batch_size, FLAGS.num_steps, 3)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    x = points
    y = targets

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model.fit(x_train, y_train,
          batch_size=FLAGS.batch_size, epochs=1, shuffle=True,
          validation_data=(x_test, y_test),callbacks=[checkpointer])
    
    model.save(FLAGS.checkpoint_dir+"/lstmsplit.h5")

def trainfrombak(points, seqlength, targets,checkpointer):
    model = load_model(FLAGS.checkpoint_dir+"/lstmsplit.h5")
    x = points
    y = targets

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model.fit(x_train, y_train,
          batch_size=FLAGS.batch_size, epochs=3, shuffle=True,
          validation_data=(x_test, y_test),callbacks=[checkpointer])

    
    # model.save(FLAGS.checkpoint_dir+"/lstmsplit.h5")

def predict(points, seqlength, targets):
    model = load_model(FLAGS.checkpoint_dir+"/lstmsplit.h5")
    
    x_test = points
    y_test = targets
    label = np.reshape(y_test,[len(y_test)])
    c = Counter(label)
    print('Resampled dataset shape {}'.format(c))
    result = model.predict(x_test,batch_size=128,verbose=1)
    poserr = 0
    negerr = 0
    pos = 0
    neg = 0
    for i in range(len(x_test)):
        

        if result[i]>0.5 and y_test[i] ==0:
            negerr += 1
        if result[i]<0.5 and y_test[i]==1:
            poserr += 1
    print(result[:10])
    print("validate result poserr:{},negerr:{}".format(poserr,negerr))
    print("accuracy posrate:{},negrate:{}".format(poserr/c[1],negerr/c[0]))

def procedata():
    datafile = open(FLAGS.dataset, 'r')
    lines = datafile.readlines()
    random.shuffle(lines)
    points = []
    maxlen = 0
    minlen = 9999
    pLength = []
    label = []
    totalseqs = FLAGS.num_steps*3
    for line in lines:
        data = line.split('\t')
        label.append([int(data[2][0])])
        bhv = data[1].split(',')[:-1]
        inst = np.zeros(totalseqs)
        one = np.array([float(i) for i in bhv])
        if len(one) <= totalseqs:
            inst[:len(one)] = one
            pLength.append(int(len(one)/3))
        else:
            inst = one[:totalseqs]
            pLength.append(FLAGS.num_steps)
        inst = np.reshape(inst,[FLAGS.num_steps,3])
        points.append(inst)
    datafile.close()
    points = np.array(points)
    pLength = np.array(pLength)
    label = np.array(label)
    print (len(points))
    return points, pLength, label

def plotmodel():
    model = load_model(FLAGS.checkpoint_dir+"/lstmsplit.h5")
    print(model.summary())
    # plot_model(model, to_file='model.png')

if __name__ == '__main__':
    checkpointer = ModelCheckpoint(filepath=FLAGS.checkpoint_dir+"/lstm-{epoch:02d}-{val_loss:.2f}.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    
    points, length, label = procedata()
    if FLAGS.mode == "train":
        
        train(points, length, label,checkpointer)
    elif FLAGS.mode == "frombak":
        
        trainfrombak(points, length, label,checkpointer)
    elif FLAGS.mode == "predict":
        
        predict(points, length, label)
    elif FLAGS.mode == "show":
        print("can not work now==============")
        plotmodel()
#
