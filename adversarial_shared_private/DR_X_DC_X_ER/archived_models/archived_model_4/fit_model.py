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

os.environ["CUDA_VISIBLE_DEVICES"]="6"





def mean_squared_error(y_true, y_pred):
	error = K.square(y_pred - y_true)
	return K.mean(K.sum(K.sum(error, axis=2), axis = 1))

def L_diff(y_true, y_pred):
	return y_pred





X_train_DW, Y_train_DW, Y_train_DW_class, X_train_gender_DW = load_DAIC_WOZ_training_data()
Y_train_DW_task = np.array([[1,0]]*X_train_DW.shape[0])						# DW = [1,0]

X_dev_DW, Y_dev_DW, Y_dev_DW_class, X_dev_gender_DW = load_DAIC_WOZ_development_data()
Y_dev_DW_task = np.array([[1,0]]*X_dev_DW.shape[0])


X_train_ER, Y_train_ER = load_CMU_MOSEI_training_data()
Y_train_ER_task = np.array([[0,1]]*X_train_ER.shape[0])						# ER = [0,1]

X_dev_ER, Y_dev_ER = load_CMU_MOSEI_development_data()
Y_dev_ER_task = np.array([[0,1]]*X_dev_ER.shape[0])


loss_funcs_DW = {'DWR_output_layer' : 'mse', 'DWC_output_layer' : 'categorical_crossentropy', 'shared_discriminator_output_layer' : 'categorical_crossentropy', 'DW_L_diff_layer' : L_diff}
loss_weights_DW = {'DWR_output_layer' : 1.09, 'DWC_output_layer' : 1.09, 'shared_discriminator_output_layer' : 0.25, 'DW_L_diff_layer' : 0.08}
metrics_DW = {'DWR_output_layer' : ['mae'], 'DWC_output_layer' : ['accuracy']}


loss_funcs_ER = {'ER_output_layer' : mean_squared_error, 'shared_discriminator_output_layer' : 'categorical_crossentropy', 'ER_L_diff_layer' : L_diff}
loss_weights_ER = {'ER_output_layer' : 1.09, 'shared_discriminator_output_layer' : 0.25, 'ER_L_diff_layer' : 0.08}
#metrics_ER = {'ER_output_layer' : ['mse', 'mae']}



DW_model = load_DAIC_WOZ_model()
DW_model.compile(optimizer = 'adam', loss = loss_funcs_DW, loss_weights = loss_weights_DW, metrics = metrics_DW)

ER_model = load_CMU_MOSEI_model()
ER_model.compile(optimizer = 'adam', loss = loss_funcs_ER, loss_weights = loss_weights_ER)



min_DR_MSE = 100000
min_DC_crossentropy = 100000
max_DC_ACC = -1
min_ER_MSE = 100000

DW_development_curve = []
ER_development_curve = []

total_epoch_count = 0

epochs_on_a_task = 40

while(total_epoch_count < 400):

	if(path.exists('ER_model_weights.h5')):
		DW_model.load_weights('ER_model_weights.h5', by_name = True)

	for epoch in range(epochs_on_a_task):
		
		DW_model.fit(X_train_DW, [Y_train_DW, Y_train_DW_class, Y_train_DW_task, np.ones((X_train_DW.shape[0],))], batch_size = X_train_DW.shape[0])
		DW_model.save_weights('DW_model_weights.h5')
		loss_dev_DW = DW_model.evaluate(X_dev_DW, [Y_dev_DW, Y_dev_DW_class, Y_dev_DW_task, np.ones((X_dev_DW.shape[0],))], batch_size = X_dev_DW.shape[0])
		
		ER_model.load_weights('DW_model_weights.h5', by_name = True)
		loss_dev_ER = ER_model.evaluate(X_dev_ER, [Y_dev_ER, Y_dev_ER_task, np.ones((X_dev_ER.shape[0],))])
		
		print('loss_dev_DW :', loss_dev_DW)
		print('loss_dev_ER :', loss_dev_ER)

		DW_development_curve.append([total_epoch_count] + loss_dev_DW)
		ER_development_curve.append([total_epoch_count] + loss_dev_ER)

		if(loss_dev_DW[1] < min_DR_MSE):
			print("DR_MSE "*200)

			min_DR_MSE = loss_dev_DW[1]
			DW_model.save_weights('optonDR_MSE_DW_weights.h5')
			ER_model.save_weights('optonDR_MSE_ER_weights.h5')

			with open('DR_MSE_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DW[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DW[5]))
				f.write('\n')
				f.write('DC_accuracy:\t'+str(loss_dev_DW[6]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('ER_MSE:\t'+str(loss_dev_ER[1]))
				f.write('\n')
				f.write('DW_total_loss:\t'+str(loss_dev_DW[0]))
				f.write('\n')
				f.write('DW_discriminator_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('DW_L_diff:\t'+str(loss_dev_DW[4]))
				f.write('\n')
				f.write('ER_total_loss:\t'+str(loss_dev_ER[0]))
				f.write('\n')
				f.write('ER_discriminator_crossentropy:\t'+str(loss_dev_ER[2]))
				f.write('\n')
				f.write('ER_L_diff:\t'+str(loss_dev_ER[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')


		if(loss_dev_DW[6] > max_DC_ACC):
			print("DC_ACC "*200)

			max_DC_ACC = loss_dev_DW[6]
			DW_model.save_weights('optonDC_ACC_DW_weights.h5')
			ER_model.save_weights('optonDC_ACC_ER_weights.h5')

			with open('DC_ACC_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DW[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DW[5]))
				f.write('\n')
				f.write('DC_accuracy:\t'+str(loss_dev_DW[6]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('ER_MSE:\t'+str(loss_dev_ER[1]))
				f.write('\n')
				f.write('DW_total_loss:\t'+str(loss_dev_DW[0]))
				f.write('\n')
				f.write('DW_discriminator_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('DW_L_diff:\t'+str(loss_dev_DW[4]))
				f.write('\n')
				f.write('ER_total_loss:\t'+str(loss_dev_ER[0]))
				f.write('\n')
				f.write('ER_discriminator_crossentropy:\t'+str(loss_dev_ER[2]))
				f.write('\n')
				f.write('ER_L_diff:\t'+str(loss_dev_ER[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')

		if(loss_dev_DW[2] < min_DC_crossentropy):
			print("DC_crossentropy "*20)

			min_DC_crossentropy = loss_dev_DW[2]

			DW_model.save_weights('optonDC_crossentropy_DW_weights.h5')
			ER_model.save_weights('optonDC_crossentropy_ER_weights.h5')

			with open('DC_crossentropy_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DW[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DW[5]))
				f.write('\n')
				f.write('DC_accuracy:\t'+str(loss_dev_DW[6]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('ER_MSE:\t'+str(loss_dev_ER[1]))
				f.write('\n')
				f.write('DW_total_loss:\t'+str(loss_dev_DW[0]))
				f.write('\n')
				f.write('DW_discriminator_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('DW_L_diff:\t'+str(loss_dev_DW[4]))
				f.write('\n')
				f.write('ER_total_loss:\t'+str(loss_dev_ER[0]))
				f.write('\n')
				f.write('ER_discriminator_crossentropy:\t'+str(loss_dev_ER[2]))
				f.write('\n')
				f.write('ER_L_diff:\t'+str(loss_dev_ER[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')


		if(loss_dev_ER[1] < min_ER_MSE):
			print("ER_MSE "*200)

			min_ER_MSE = loss_dev_ER[1]

			DW_model.save_weights('optonER_MSE_DW_weights.h5')
			ER_model.save_weights('optonER_MSE_ER_weights.h5')

			with open('ER_MSE_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DW[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DW[5]))
				f.write('\n')
				f.write('DC_accuracy:\t'+str(loss_dev_DW[6]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('ER_MSE:\t'+str(loss_dev_ER[1]))
				f.write('\n')
				f.write('DW_total_loss:\t'+str(loss_dev_DW[0]))
				f.write('\n')
				f.write('DW_discriminator_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('DW_L_diff:\t'+str(loss_dev_DW[4]))
				f.write('\n')
				f.write('ER_total_loss:\t'+str(loss_dev_ER[0]))
				f.write('\n')
				f.write('ER_discriminator_crossentropy:\t'+str(loss_dev_ER[2]))
				f.write('\n')
				f.write('ER_L_diff:\t'+str(loss_dev_ER[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')

		total_epoch_count = total_epoch_count + 1

	DW_model.save_weights('DW_model_weights.h5')

	np.save('DW_development_progress.npy', np.array(DW_development_curve))
	np.save('ER_development_progress.npy', np.array(ER_development_curve))


	print('\n\n')
	print("="*400)
	print("="*400)
	print('\n\n')


	if(path.exists('DW_model_weights.h5')):
		ER_model.load_weights('DW_model_weights.h5', by_name = True)


	for epoch in range(epochs_on_a_task):
		
		ER_model.fit(X_train_ER, [Y_train_ER, Y_train_ER_task, np.ones((X_train_ER.shape[0],))], batch_size = X_train_ER.shape[0])
		ER_model.save_weights('ER_model_weights.h5')
		loss_dev_ER = ER_model.evaluate(X_dev_ER, [Y_dev_ER, Y_dev_ER_task, np.ones((X_dev_ER.shape[0],))])

		DW_model.load_weights('ER_model_weights.h5', by_name = True)
		loss_dev_DW = DW_model.evaluate(X_dev_DW, [Y_dev_DW, Y_dev_DW_class, Y_dev_DW_task, np.ones((X_dev_DW.shape[0],))])
		
		print('loss_dev_DW :', loss_dev_DW)
		print('loss_dev_ER :', loss_dev_ER)

		DW_development_curve.append([total_epoch_count] + loss_dev_DW)
		ER_development_curve.append([total_epoch_count] + loss_dev_ER)

		if(loss_dev_DW[1] < min_DR_MSE):
			print("DR_MSE "*200)

			min_DR_MSE = loss_dev_DW[1]
			DW_model.save_weights('optonDR_MSE_DW_weights.h5')
			ER_model.save_weights('optonDR_MSE_ER_weights.h5')

			with open('DR_MSE_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DW[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DW[5]))
				f.write('\n')
				f.write('DC_accuracy:\t'+str(loss_dev_DW[6]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('ER_MSE:\t'+str(loss_dev_ER[1]))
				f.write('\n')
				f.write('DW_total_loss:\t'+str(loss_dev_DW[0]))
				f.write('\n')
				f.write('DW_discriminator_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('DW_L_diff:\t'+str(loss_dev_DW[4]))
				f.write('\n')
				f.write('ER_total_loss:\t'+str(loss_dev_ER[0]))
				f.write('\n')
				f.write('ER_discriminator_crossentropy:\t'+str(loss_dev_ER[2]))
				f.write('\n')
				f.write('ER_L_diff:\t'+str(loss_dev_ER[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')


		if(loss_dev_DW[6] > max_DC_ACC):
			print("DC_ACC "*200)

			max_DC_ACC = loss_dev_DW[6]
			DW_model.save_weights('optonDC_ACC_DW_weights.h5')
			ER_model.save_weights('optonDC_ACC_ER_weights.h5')

			with open('DC_ACC_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DW[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DW[5]))
				f.write('\n')
				f.write('DC_accuracy:\t'+str(loss_dev_DW[6]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('ER_MSE:\t'+str(loss_dev_ER[1]))
				f.write('\n')
				f.write('DW_total_loss:\t'+str(loss_dev_DW[0]))
				f.write('\n')
				f.write('DW_discriminator_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('DW_L_diff:\t'+str(loss_dev_DW[4]))
				f.write('\n')
				f.write('ER_total_loss:\t'+str(loss_dev_ER[0]))
				f.write('\n')
				f.write('ER_discriminator_crossentropy:\t'+str(loss_dev_ER[2]))
				f.write('\n')
				f.write('ER_L_diff:\t'+str(loss_dev_ER[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')


		if(loss_dev_DW[2] < min_DC_crossentropy):
			print("DC_crossentropy "*20)

			min_DC_crossentropy = loss_dev_DW[2]

			DW_model.save_weights('optonDC_crossentropy_DW_weights.h5')
			ER_model.save_weights('optonDC_crossentropy_ER_weights.h5')

			with open('DC_crossentropy_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DW[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DW[5]))
				f.write('\n')
				f.write('DC_accuracy:\t'+str(loss_dev_DW[6]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('ER_MSE:\t'+str(loss_dev_ER[1]))
				f.write('\n')
				f.write('DW_total_loss:\t'+str(loss_dev_DW[0]))
				f.write('\n')
				f.write('DW_discriminator_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('DW_L_diff:\t'+str(loss_dev_DW[4]))
				f.write('\n')
				f.write('ER_total_loss:\t'+str(loss_dev_ER[0]))
				f.write('\n')
				f.write('ER_discriminator_crossentropy:\t'+str(loss_dev_ER[2]))
				f.write('\n')
				f.write('ER_L_diff:\t'+str(loss_dev_ER[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')


		if(loss_dev_ER[1] < min_ER_MSE):
			print("ER_MSE "*200)

			min_ER_MSE = loss_dev_ER[1]

			DW_model.save_weights('optonER_MSE_DW_weights.h5')
			ER_model.save_weights('optonER_MSE_ER_weights.h5')

			with open('ER_MSE_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DW[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DW[5]))
				f.write('\n')
				f.write('DC_accuracy:\t'+str(loss_dev_DW[6]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('ER_MSE:\t'+str(loss_dev_ER[1]))
				f.write('\n')
				f.write('DW_total_loss:\t'+str(loss_dev_DW[0]))
				f.write('\n')
				f.write('DW_discriminator_crossentropy:\t'+str(loss_dev_DW[2]))
				f.write('\n')
				f.write('DW_L_diff:\t'+str(loss_dev_DW[4]))
				f.write('\n')
				f.write('ER_total_loss:\t'+str(loss_dev_ER[0]))
				f.write('\n')
				f.write('ER_discriminator_crossentropy:\t'+str(loss_dev_ER[2]))
				f.write('\n')
				f.write('ER_L_diff:\t'+str(loss_dev_ER[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')

		total_epoch_count = total_epoch_count + 1

	ER_model.save_weights('ER_model_weights.h5')

	np.save('DW_development_progress.npy', np.array(DW_development_curve))
	np.save('ER_development_progress.npy', np.array(ER_development_curve))


	print('\n\n')
	print("="*400)
	print("="*400)
	print('\n\n')