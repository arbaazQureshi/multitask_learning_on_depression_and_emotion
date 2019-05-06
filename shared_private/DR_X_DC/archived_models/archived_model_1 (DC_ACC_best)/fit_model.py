from keras.models import Model, load_model

from load_model import load_DR_model, load_DC_model
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


DR_model = load_DR_model()
DR_model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])

DC_model = load_DC_model()
DC_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


X_train_DW, Y_train_DW, Y_train_DW_class, X_train_gender_DW = load_DAIC_WOZ_training_data()
X_dev_DW, Y_dev_DW, Y_dev_DW_class, X_dev_gender_DW = load_DAIC_WOZ_development_data()


mini2 = -1
mini1 = 100000

DR_development_curve = []
DC_development_curve = []

total_epoch_count = 0

while(total_epoch_count < 500):

	if(path.exists('DC_model_weights.h5')):
		DC_model.load_weights('DC_model_weights.h5', by_name = True)

	for epoch in range(25):
		
		DR_model.fit(X_train_DW, Y_train_DW, batch_size = X_train_DW.shape[0])
		DR_model.save_weights('DR_model_weights.h5')
		loss_dev_DR = DR_model.evaluate(X_dev_DW, Y_dev_DW)
		
		DC_model.load_weights('DR_model_weights.h5', by_name = True)
		loss_dev_DC = DC_model.evaluate(X_dev_DW, Y_dev_DW_class)
		
		DR_development_curve.append([total_epoch_count, loss_dev_DR[0], loss_dev_DR[1]])
		DC_development_curve.append([total_epoch_count, loss_dev_DC[0], loss_dev_DC[1]])

		if(loss_dev_DR[0] < mini1):
			print("DR_MSE "*200)

			mini1 = loss_dev_DR[0]

			DR_model.save_weights('optonDR_DR_weights.h5')
			DC_model.save_weights('optonDR_DC_weights.h5')

			with open('DR_MSE_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('DC_loss:\t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))

		if(loss_dev_DC[1] > mini2):
			print("DC_ACC "*200)

			mini2 = loss_dev_DC[1]

			DR_model.save_weights('optonDC_DR_weights.h5')
			DC_model.save_weights('optonDC_DC_weights.h5')

			with open('DC_ACC_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('DC_loss:\t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))

		total_epoch_count = total_epoch_count + 1

	DC_model.save_weights('DC_model_weights.h5')











	if(path.exists('DC_model_weights.h5')):
		DC_model.load_weights('DC_model_weights.h5', by_name = True)


	for epoch in range(25):
		
		DC_model.fit(X_train_DW, Y_train_DW_class, batch_size = X_train_DW.shape[0])
		DC_model.save_weights('DC_model_weights.h5')
		loss_dev_DC = DC_model.evaluate(X_dev_DW, Y_dev_DW_class)

		DR_model.load_weights('DC_model_weights.h5', by_name = True)
		loss_dev_DR = DR_model.evaluate(X_dev_DW, Y_dev_DW)
		
		DR_development_curve.append([total_epoch_count, loss_dev_DR[0], loss_dev_DR[1]])
		DC_development_curve.append([total_epoch_count, loss_dev_DC[0], loss_dev_DC[1]])


		if(loss_dev_DR[0] < mini1):
			print("DR_MSE "*200)

			mini1 = loss_dev_DR[0]

			DR_model.save_weights('optonDR_DR_weights.h5')
			DC_model.save_weights('optonDR_DC_weights.h5')

			with open('DR_MSE_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('DC_loss:\t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))

		if(loss_dev_DC[1] > mini2):
			print("DC_ACC "*200)

			mini2 = loss_dev_DC[1]

			DR_model.save_weights('optonDC_DR_weights.h5')
			DC_model.save_weights('optonDC_DC_weights.h5')

			with open('DC_ACC_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('DC_loss:\t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))

		total_epoch_count = total_epoch_count + 1

	DC_model.save_weights('DC_model_weights.h5')

	np.save('DR_development_progress.npy', np.array(DR_development_curve))
	np.save('DC_development_progress.npy', np.array(DC_development_curve))