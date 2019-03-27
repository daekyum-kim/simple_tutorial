from __future__ import print_function
import keras
import os, sys, time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint

def input():
  x_train = np.array([[1, 2, 3, 4], 
    [5, 6, 7 ,8], 
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
    [21, 22, 23, 24],
    [25, 26, 27, 28],
    [29, 30, 31, 32],
    [45, 46, 47, 48],
    [49, 50, 51, 52]])
  y_train = np.array([40, 96, 152, 208, 264, 320, 376, 432, 656, 712])

  x_test = np.array([[33, 34, 35, 36], [37, 38, 39 ,40], [41, 42, 43, 44]])
  y_test = np.array([488, 544, 600])

  x_train = x_train.astype('float32')
  y_train = y_train.astype('float32')

  x_test = x_test.astype('float32')
  y_test = y_test.astype('float32')

  for i in range(0, x_train.shape[0]):
    x_train[i,:]/=np.max(x_train[i,:])
  for i in range(0, x_test.shape[0]):
    x_test[i,:]/=np.max(x_test[i,:])

  np.reshape(x_train, (-1, x_train.shape[1], 1))
  np.reshape(x_test, (-1, x_test.shape[1], 1))

  return x_train, x_test, y_train, y_test



def layers(shape_in):
  model = Sequential()
  model.add(Dense(128, input_shape=(shape_in, ), activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(1, activation='linear'))

  sgd_ = SGD(lr=0.001, decay=1e-6, momentum=1.0)
  rmsprop_ = RMSprop(lr = 0.001, rho = 0.9, decay=1e-6)
  adam_ = Adam(lr = 0.001, decay=1e-6)

  model.compile(loss='mean_squared_error',
    optimizer=adam_,
    metrics=['mae', 'mse'])
  return model


if __name__ == '__main__':
  batch_size = 2
  epochs = 7000
  start = time.time()

  x_train, x_test, y_train, y_test = input()
  
  model = layers(x_train.shape[1])
  filepath="weights.best.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]

  model.fit(x_train, y_train, callbacks=callbacks_list,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2)
 #training finished, and prediction
  result = model.predict(x_test)
  end = time.time()
  print("ideal result: \n", y_test)
  print("obtained result: \n", result)
  score = model.evaluate(x_test, y_test, verbose=0)
  print('mean_absolute_error:', score[1])
  print('elapsed_time: %.2f seconds' % (end - start))
  # print('Test loss:', score[0])