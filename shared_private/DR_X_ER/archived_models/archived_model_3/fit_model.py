from keras.models import Model, load_model

from load_model import load_DAIC_WOZ_model, load_CMU_MOSEI_model
from load_data import load_DAIC_WOZ_training_data, load_DAIC_WOZ_development_data, load_CMU_MOSEI_training_data, load_CMU_MOSEI_development_data
import keras
import keras.backend as K

import numpy as np
import os
from os import path

import random

os.environ["CUDA_VISIBLE_DEVICES"]="5"



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


DW_model = load_DAIC_WOZ_model()
DW_model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])

CM_model = load_CMU_MOSEI_model()
CM_model.compile(optimizer = 'adam', loss = mean_squared_error)


X_train_DW, Y_train_DW, X_train_gender_DW = load_DAIC_WOZ_training_data()

#X_train_DW = np.array(X_train_DW)
#X_train_gender_DW = np.array(X_train_gender_DW)

X_dev_DW, Y_dev_DW, X_dev_gender_DW = load_DAIC_WOZ_development_data()

#X_dev_DW = np.array(X_dev_DW)
#X_dev_gender_DW = np.array(X_dev_gender_DW)


X_train_CM, Y_train_CM = load_CMU_MOSEI_training_data()
#X_train_CM = np.array(X_train_CM)

X_dev_CM, Y_dev_CM = load_CMU_MOSEI_development_data()
#X_dev_CM = np.array(X_dev_CM)



mini1 = 100000
mini2 = 100000

DW_development_curve = []
CM_development_curve = []

total_epoch_count = 0

while(True):

	if(path.exists('CM_model_weights.h5')):
		DW_model.load_weights('CM_model_weights.h5', by_name = True)

	for epoch in range(25):
		
		DW_model.fit(X_train_DW, Y_train_DW, batch_size = X_train_DW.shape[0])
		DW_model.save_weights('DW_model_weights.h5')
		loss_dev_DW = DW_model.evaluate(X_dev_DW, Y_dev_DW)
		
		CM_model.load_weights('DW_model_weights.h5', by_name = True)
		loss_dev_CM = CM_model.evaluate(X_dev_CM, Y_dev_CM)
		
		DW_development_curve.append([total_epoch_count, loss_dev_DW[0], loss_dev_DW[1]])
		CM_development_curve.append([total_epoch_count, loss_dev_CM])

		if(loss_dev_DW[0] < mini1):
			print("* "*2500)

			mini1 = loss_dev_DW[0]
			DW_model.save_weights('optonDW_DW_weights.h5')
			CM_model.save_weights('optonDW_CM_weights.h5')

			with open('DW_current_best.txt', 'w') as f:
				f.write(str(loss_dev_DW[0]))
				f.write('\t')
				f.write(str(loss_dev_DW[1]))

		if(loss_dev_CM < mini2):
			print("| "*2500)

			mini2 = loss_dev_CM

			DW_model.save_weights('optonCM_DW_weights.h5')
			CM_model.save_weights('optonCM_CM_weights.h5')

			with open('CM_current_best.txt', 'w') as f:
				f.write(str(loss_dev_CM))

		total_epoch_count = total_epoch_count + 1

	DW_model.save_weights('DW_model_weights.h5')











	if(path.exists('DW_model_weights.h5')):
		CM_model.load_weights('DW_model_weights.h5', by_name = True)


	for epoch in range(25):
		
		CM_model.fit(X_train_CM, Y_train_CM, batch_size = X_train_CM.shape[0])
		CM_model.save_weights('CM_model_weights.h5')
		loss_dev_CM = CM_model.evaluate(X_dev_CM, Y_dev_CM)

		DW_model.load_weights('CM_model_weights.h5', by_name = True)
		loss_dev_DW = DW_model.evaluate(X_dev_DW, Y_dev_DW)
		
		DW_development_curve.append([total_epoch_count, loss_dev_DW[0], loss_dev_DW[1]])
		CM_development_curve.append([total_epoch_count, loss_dev_CM])


		if(loss_dev_DW[0] < mini1):
			print("* "*2500)

			mini1 = loss_dev_DW[0]
			DW_model.save_weights('optonDW_DW_weights.h5')
			CM_model.save_weights('optonDW_CM_weights.h5')

			with open('DW_current_best.txt', 'w') as f:
				f.write(str(loss_dev_DW[0]))
				f.write('\t')
				f.write(str(loss_dev_DW[1]))

		if(loss_dev_CM < mini2):
			print("| "*2500)

			mini2 = loss_dev_CM

			DW_model.save_weights('optonCM_DW_weights.h5')
			CM_model.save_weights('optonCM_CM_weights.h5')

			with open('CM_current_best.txt', 'w') as f:
				f.write(str(loss_dev_CM))

		total_epoch_count = total_epoch_count + 1

	CM_model.save_weights('CM_model_weights.h5')

	np.save('DW_development_progress.npy', np.array(DW_development_curve))
	np.save('CM_development_progress.npy', np.array(CM_development_curve))