from keras.models import Model, load_model

from load_model import load_CMU_MOSEI_model
from load_data import load_CMU_MOSEI_training_data, load_CMU_MOSEI_development_data
import keras
import keras.backend as K

import numpy as np
import os
from os import path

import random

os.environ["CUDA_VISIBLE_DEVICES"]="1"



def mean_squared_error(y_true, y_pred):
	error = K.square(y_pred - y_true)
	return K.mean(K.sum(K.sum(error, axis=2), axis = 1))



#if(path.exists('training_progress.csv')):
#	progress = np.loadtxt('training_progress.csv', delimiter=',').tolist()

#else:
progress = []

#if(path.exists('optimal_weights.h5')):
#	model = load_model()
#	model.load_weights('optimal_weights.h5')

#else:
#	model = load_model()
#	model.compile(optimizer='adam', loss='mse', metrics = ['mae'])


CM_model = load_CMU_MOSEI_model()
CM_model.compile(optimizer = 'adam', loss = mean_squared_error, metrics = ['mae'])

X_train_CM, Y_train_CM = load_CMU_MOSEI_training_data()
X_dev_CM, Y_dev_CM = load_CMU_MOSEI_development_data()



mini = 100000


CM_training_curve = []
CM_development_curve = []


total_epoch_count = 0

while(True):

	hist = CM_model.fit(X_train_CM, Y_train_CM, batch_size = X_train_CM.shape[0])
	loss_train = hist.history['loss'][-1]
	loss_dev = CM_model.evaluate(X_dev_CM, Y_dev_CM)
	print(loss_dev)

	CM_training_curve.append([loss_train, total_epoch_count])
	CM_development_curve.append([loss_dev[0], loss_dev[1], total_epoch_count])

	if(loss_dev[0] < mini):
		print("*"*5000)
		mini = loss_dev[0]
		CM_model.save('best_model.h5')

		with open('BEST_TILL_NOW.txt', 'w') as f:
			f.write(str(loss_dev[0]))
			f.write('\t')
			f.write(str(loss_dev[1]))

	total_epoch_count = total_epoch_count + 1

	np.save('CM_development_progress.npy', np.array(CM_development_curve))
	np.save('CM_training_progress.npy', np.array(CM_training_curve))