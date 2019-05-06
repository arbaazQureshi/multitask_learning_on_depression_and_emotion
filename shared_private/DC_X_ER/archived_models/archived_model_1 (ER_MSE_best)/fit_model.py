from keras.models import Model, load_model

from load_model import load_DC_model, load_ER_model
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


DC_model = load_DC_model()
DC_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

CM_model = load_ER_model()
CM_model.compile(optimizer = 'adam', loss = mean_squared_error)


X_train_DW, Y_train_DW, X_train_gender_DW = load_DAIC_WOZ_training_data()
X_dev_DW, Y_dev_DW, X_dev_gender_DW = load_DAIC_WOZ_development_data()

X_train_CM, Y_train_CM = load_CMU_MOSEI_training_data()
X_dev_CM, Y_dev_CM = load_CMU_MOSEI_development_data()



mini1_loss = 100000
mini1_accuracy = -1
mini2 = 100000

DW_development_curve = []
CM_development_curve = []

total_epoch_count = 0

while(total_epoch_count < 500):

	if(path.exists('CM_model_weights.h5')):
		DC_model.load_weights('CM_model_weights.h5', by_name = True)

	for epoch in range(25):
		
		DC_model.fit(X_train_DW, Y_train_DW, batch_size = X_train_DW.shape[0])
		DC_model.save_weights('DC_model_weights.h5')
		loss_dev_DW = DC_model.evaluate(X_dev_DW, Y_dev_DW)
		
		CM_model.load_weights('DC_model_weights.h5', by_name = True)
		loss_dev_CM = CM_model.evaluate(X_dev_CM, Y_dev_CM)
		
		DW_development_curve.append([total_epoch_count, loss_dev_DW[0], loss_dev_DW[1]])
		CM_development_curve.append([total_epoch_count, loss_dev_CM])

		if(loss_dev_DW[1] > mini1_accuracy):
			print("DC_ACC "*200)

			mini1_accuracy = loss_dev_DW[1]

			DC_model.save_weights('optonDC_DC_weights.h5')
			CM_model.save_weights('optonDC_CM_weights.h5')

			with open('DC_acc_current_best.txt', 'w') as f:
				f.write('DC_loss:\t'+str(loss_dev_DW[0]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DW[1]))
				f.write('\n')
				f.write('ER_MSE:\t'+str(loss_dev_CM))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))

		if(loss_dev_CM < mini2):
			print("ER_MSE "*200)

			mini2 = loss_dev_CM

			DC_model.save_weights('optonCM_DW_weights.h5')
			CM_model.save_weights('optonCM_CM_weights.h5')

			with open('ER_MSE_current_best.txt', 'w') as f:
				f.write('DC_loss:\t'+str(loss_dev_DW[0]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DW[1]))
				f.write('\n')
				f.write('ER_MSE:\t'+str(loss_dev_CM))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))

		total_epoch_count = total_epoch_count + 1

	DC_model.save_weights('DC_model_weights.h5')











	if(path.exists('DC_model_weights.h5')):
		CM_model.load_weights('DC_model_weights.h5', by_name = True)


	for epoch in range(25):
		
		CM_model.fit(X_train_CM, Y_train_CM, batch_size = X_train_CM.shape[0])
		CM_model.save_weights('CM_model_weights.h5')
		loss_dev_CM = CM_model.evaluate(X_dev_CM, Y_dev_CM)

		DC_model.load_weights('CM_model_weights.h5', by_name = True)
		loss_dev_DW = DC_model.evaluate(X_dev_DW, Y_dev_DW)
		
		DW_development_curve.append([total_epoch_count, loss_dev_DW[0], loss_dev_DW[1]])
		CM_development_curve.append([total_epoch_count, loss_dev_CM])


		if(loss_dev_DW[1] > mini1_accuracy):
			print("DC_ACC "*200)

			mini1_accuracy = loss_dev_DW[1]

			DC_model.save_weights('optonDC_DC_weights.h5')
			CM_model.save_weights('optonDC_CM_weights.h5')

			with open('DC_acc_current_best.txt', 'w') as f:
				f.write('DC_loss:\t'+str(loss_dev_DW[0]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DW[1]))
				f.write('\n')
				f.write('ER_MSE:\t'+str(loss_dev_CM))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))

		if(loss_dev_CM < mini2):
			print("ER_MSE "*200)

			mini2 = loss_dev_CM

			DC_model.save_weights('optonCM_DW_weights.h5')
			CM_model.save_weights('optonCM_CM_weights.h5')

			with open('ER_MSE_current_best.txt', 'w') as f:
				f.write('DC_loss:\t'+str(loss_dev_DW[0]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DW[1]))
				f.write('\n')
				f.write('ER_MSE:\t'+str(loss_dev_CM))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))

		total_epoch_count = total_epoch_count + 1

	CM_model.save_weights('CM_model_weights.h5')

	np.save('DW_development_progress.npy', np.array(DW_development_curve))
	np.save('CM_development_progress.npy', np.array(CM_development_curve))