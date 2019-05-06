from keras.models import Model, load_model

from load_model import load_DAIC_WOZ_model, load_CMU_MOSEI_model
from load_data import load_DAIC_WOZ_training_data, load_DAIC_WOZ_development_data, load_CMU_MOSEI_training_data, load_CMU_MOSEI_development_data
import keras
import keras.backend as K

from keras.losses import categorical_crossentropy

import numpy as np
import os
from os import path

import random

os.environ["CUDA_VISIBLE_DEVICES"]="4"





def mean_squared_error(y_true, y_pred):
	error = K.square(y_pred - y_true)
	return K.mean(K.sum(K.sum(error, axis=2), axis = 1))

def L_diff(y_true, y_pred):
	return y_pred





X_train_DW, Y_train_DW, X_train_gender_DW = load_DAIC_WOZ_training_data()
Y_train_DW_task = np.array([[1,0]]*X_train_DW.shape[0])						# DW = [1,0]

X_dev_DW, Y_dev_DW, X_dev_gender_DW = load_DAIC_WOZ_development_data()
Y_dev_DW_task = np.array([[1,0]]*X_dev_DW.shape[0])


X_train_CM, Y_train_CM = load_CMU_MOSEI_training_data()
Y_train_CM_task = np.array([[0,1]]*X_train_CM.shape[0])						# CM = [0,1]

X_dev_CM, Y_dev_CM = load_CMU_MOSEI_development_data()
Y_dev_CM_task = np.array([[0,1]]*X_dev_CM.shape[0])


loss_funcs_DW = {'DW_output_layer' : 'mse', 'shared_discriminator_output_layer' : 'categorical_crossentropy', 'DW_L_diff_layer' : L_diff}
loss_weights_DW = {'DW_output_layer' : 1.09, 'shared_discriminator_output_layer' : 0.25, 'DW_L_diff_layer' : 0.08}
metrics_DW = {'DW_output_layer' : ['mse', 'mae']}


loss_funcs_CM = {'CM_output_layer' : mean_squared_error, 'shared_discriminator_output_layer' : 'categorical_crossentropy', 'CM_L_diff_layer' : L_diff}
loss_weights_CM = {'CM_output_layer' : 1.09, 'shared_discriminator_output_layer' : 0.25, 'CM_L_diff_layer' : 0.08}
metrics_CM = {'DW_output_layer' : ['mse', 'mae']}



DW_model = load_DAIC_WOZ_model()
DW_model.compile(optimizer = 'adam', loss = loss_funcs_DW, loss_weights = loss_weights_DW, metrics = metrics_DW)

CM_model = load_CMU_MOSEI_model()
CM_model.compile(optimizer = 'adam', loss = loss_funcs_CM, loss_weights = loss_weights_CM, metrics = metrics_CM)



mini1 = 100000
mini2 = 100000

DW_development_curve = []
CM_development_curve = []

total_epoch_count = 0

epochs_on_a_task = 40

while(True):

	if(path.exists('CM_model_weights.h5')):
		DW_model.load_weights('CM_model_weights.h5', by_name = True)

	for epoch in range(epochs_on_a_task):
		
		DW_model.fit(X_train_DW, [Y_train_DW, Y_train_DW_task, np.ones((X_train_DW.shape[0],))], batch_size = X_train_DW.shape[0])
		DW_model.save_weights('DW_model_weights.h5')
		loss_dev_DW = DW_model.evaluate(X_dev_DW, [Y_dev_DW, Y_dev_DW_task, np.ones((X_dev_DW.shape[0],))], batch_size = X_dev_DW.shape[0])
		
		CM_model.load_weights('DW_model_weights.h5', by_name = True)
		loss_dev_CM = CM_model.evaluate(X_dev_CM, [Y_dev_CM, Y_dev_CM_task, np.ones((X_dev_CM.shape[0],))])
		
		print('loss_dev_DW :', loss_dev_DW)
		print('loss_dev_CM :', loss_dev_CM)

		DW_development_curve.append([total_epoch_count] + loss_dev_DW)
		CM_development_curve.append([total_epoch_count] + loss_dev_CM)

		if(loss_dev_DW[4] < mini1):
			print("* "*1250)

			mini1 = loss_dev_DW[4]
			DW_model.save_weights('DW_task_optimum_DW_weights.h5')
			CM_model.save_weights('DW_task_optimum_CM_weights.h5')

			with open('DW_current_best.txt', 'w') as f:
				f.write(str(loss_dev_DW[4]))
				f.write('\t')
				f.write(str(loss_dev_DW[5]))
				f.write('\t')
				f.write(str(int(total_epoch_count)))

		if(loss_dev_CM[1] < mini2):
			print("| "*1250)

			mini2 = loss_dev_CM[1]

			DW_model.save_weights('CM_task_optimum_DW_weights.h5')
			CM_model.save_weights('CM_task_optimum_CM_weights.h5')

			with open('CM_current_best.txt', 'w') as f:
				f.write(str(loss_dev_CM[1]))
				f.write('\t')
				f.write(str(int(total_epoch_count)))

		total_epoch_count = total_epoch_count + 1

	DW_model.save_weights('DW_model_weights.h5')

	np.save('DW_development_progress.npy', np.array(DW_development_curve))
	np.save('CM_development_progress.npy', np.array(CM_development_curve))


	print('\n\n')
	print("="*500)
	print("="*500)
	print('\n\n')


	if(path.exists('DW_model_weights.h5')):
		CM_model.load_weights('DW_model_weights.h5', by_name = True)


	for epoch in range(epochs_on_a_task):
		
		CM_model.fit(X_train_CM, [Y_train_CM, Y_train_CM_task, np.ones((X_train_CM.shape[0],))], batch_size = X_train_CM.shape[0])
		CM_model.save_weights('CM_model_weights.h5')
		loss_dev_CM = CM_model.evaluate(X_dev_CM, [Y_dev_CM, Y_dev_CM_task, np.ones((X_dev_CM.shape[0],))])

		DW_model.load_weights('CM_model_weights.h5', by_name = True)
		loss_dev_DW = DW_model.evaluate(X_dev_DW, [Y_dev_DW, Y_dev_DW_task, np.ones((X_dev_DW.shape[0],))])
		
		print('loss_dev_DW :', loss_dev_DW)
		print('loss_dev_CM :', loss_dev_CM)

		DW_development_curve.append([total_epoch_count] + loss_dev_DW)
		CM_development_curve.append([total_epoch_count] + loss_dev_CM)

		if(loss_dev_DW[4] < mini1):
			print("* "*1250)

			mini1 = loss_dev_DW[4]
			DW_model.save_weights('DW_task_optimum_DW_weights.h5')
			CM_model.save_weights('DW_task_optimum_CM_weights.h5')

			with open('DW_current_best.txt', 'w') as f:
				f.write(str(loss_dev_DW[4]))
				f.write('\t')
				f.write(str(loss_dev_DW[5]))
				f.write('\t')
				f.write(str(total_epoch_count))

		if(loss_dev_CM[1] < mini2):
			print("| "*1250)

			mini2 = loss_dev_CM[1]

			DW_model.save_weights('CM_task_optimum_DW_weights.h5')
			CM_model.save_weights('CM_task_optimum_CM_weights.h5')

			with open('CM_current_best.txt', 'w') as f:
				f.write(str(loss_dev_CM[1]))
				f.write('\t')
				f.write(str(total_epoch_count))

		total_epoch_count = total_epoch_count + 1

	CM_model.save_weights('CM_model_weights.h5')

	np.save('DW_development_progress.npy', np.array(DW_development_curve))
	np.save('CM_development_progress.npy', np.array(CM_development_curve))