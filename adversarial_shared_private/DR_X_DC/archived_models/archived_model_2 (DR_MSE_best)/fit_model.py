from keras.models import Model, load_model

from load_model import load_DR_model, load_DC_model
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





X_train_D, Y_train_D, Y_train_D_class, X_train_gender_D = load_DAIC_WOZ_training_data()
Y_train_DR_task = np.array([[1,0]]*X_train_D.shape[0])
Y_train_DC_task = np.array([[0,1]]*X_train_D.shape[0])						# DR = [1,0]

X_dev_D, Y_dev_D, Y_dev_D_class, X_dev_gender_D = load_DAIC_WOZ_development_data()
Y_dev_DR_task = np.array([[1,0]]*X_dev_D.shape[0])
Y_dev_DC_task = np.array([[0,1]]*X_dev_D.shape[0])


#X_train_DC, Y_train_DC = load_CMU_MOSEI_training_data()
#Y_train_DC_task = np.array([[0,1]]*X_train_DC.shape[0])						# DC = [0,1]

#X_dev_DC, Y_dev_DC = load_CMU_MOSEI_development_data()
#Y_dev_DC_task = np.array([[0,1]]*X_dev_DC.shape[0])


loss_funcs_DR = {'DR_output_layer' : 'mse', 'shared_discriminator_output_layer' : 'categorical_crossentropy', 'DR_L_diff_layer' : L_diff}
loss_weights_DR = {'DR_output_layer' : 1.09, 'shared_discriminator_output_layer' : 0.25, 'DR_L_diff_layer' : 0.08}
metrics_DR = {'DR_output_layer' : ['mae']}


loss_funcs_DC = {'DC_output_layer' : 'categorical_crossentropy', 'shared_discriminator_output_layer' : 'categorical_crossentropy', 'DC_L_diff_layer' : L_diff}
loss_weights_DC = {'DC_output_layer' : 1.09, 'shared_discriminator_output_layer' : 0.25, 'DC_L_diff_layer' : 0.08}
metrics_DC = {'DC_output_layer' : ['accuracy']}



DR_model = load_DR_model()
DR_model.compile(optimizer = 'adam', loss = loss_funcs_DR, loss_weights = loss_weights_DR, metrics = metrics_DR)

DC_model = load_DC_model()
DC_model.compile(optimizer = 'adam', loss = loss_funcs_DC, loss_weights = loss_weights_DC, metrics = metrics_DC)



min_DR_MSE = 100000
min_DC_crossentropy = 100000
max_DC_ACC = -1

DR_development_curve = []
DC_development_curve = []

total_epoch_count = 0

epochs_on_a_task = 40

while(total_epoch_count < 400):

	if(path.exists('DC_model_weights.h5')):
		DR_model.load_weights('DC_model_weights.h5', by_name = True)

	for epoch in range(epochs_on_a_task):
		
		DR_model.fit(X_train_D, [Y_train_D, Y_train_DR_task, np.ones((X_train_D.shape[0],))], batch_size = X_train_D.shape[0])
		DR_model.save_weights('DR_model_weights.h5')
		loss_dev_DR = DR_model.evaluate(X_dev_D, [Y_dev_D, Y_dev_DR_task, np.ones((X_dev_D.shape[0],))], batch_size = X_dev_D.shape[0])
		
		DC_model.load_weights('DR_model_weights.h5', by_name = True)
		loss_dev_DC = DC_model.evaluate(X_dev_D, [Y_dev_D_class, Y_dev_DC_task, np.ones((X_dev_D.shape[0],))])
		
		print('loss_dev_DR :', loss_dev_DR)
		print('loss_dev_DC :', loss_dev_DC)

		DR_development_curve.append([total_epoch_count] + loss_dev_DR)
		DC_development_curve.append([total_epoch_count] + loss_dev_DC)

		if(loss_dev_DR[1] < min_DR_MSE):
			print("DR_MSE "*200)

			min_DR_MSE = loss_dev_DR[1]
			DR_model.save_weights('optonDR_MSE_DR_weights.h5')
			DC_model.save_weights('optonDR_MSE_DC_weights.h5')

			with open('DR_MSE_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DR[4]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DC[4]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('DR_total_loss:\t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_discriminator_crossentropy:\t'+str(loss_dev_DR[2]))
				f.write('\n')
				f.write('DR_L_diff:\t'+str(loss_dev_DR[3]))
				f.write('\n')
				f.write('DC_total_loss:\t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('DC_discriminator_crossentropy:\t'+str(loss_dev_DC[2]))
				f.write('\n')
				f.write('DC_L_diff:\t'+str(loss_dev_DC[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')


		if(loss_dev_DC[4] > max_DC_ACC):
			print("DC_ACC "*200)

			max_DC_ACC = loss_dev_DC[4]
			DR_model.save_weights('optonDC_ACC_DR_weights.h5')
			DC_model.save_weights('optonDC_ACC_DC_weights.h5')

			with open('DC_ACC_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DR[4]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DC[4]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('DR_total_loss:\t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_discriminator_crossentropy:\t'+str(loss_dev_DR[2]))
				f.write('\n')
				f.write('DR_L_diff:\t'+str(loss_dev_DR[3]))
				f.write('\n')
				f.write('DC_total_loss:\t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('DC_discriminator_crossentropy:\t'+str(loss_dev_DC[2]))
				f.write('\n')
				f.write('DC_L_diff:\t'+str(loss_dev_DC[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')
		
		if(loss_dev_DC[1] < min_DC_crossentropy):
			print("DC_crossentropy "*20)

			min_DC_crossentropy = loss_dev_DR[1]

			DR_model.save_weights('optonDC_crossentropy_DR_weights.h5')
			DC_model.save_weights('optonDC_crossentropy_DC_weights.h5')

			with open('DC_crossentropy_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DR[4]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DC[4]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('DR_total_loss:\t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_discriminator_crossentropy:\t'+str(loss_dev_DR[2]))
				f.write('\n')
				f.write('DR_L_diff:\t'+str(loss_dev_DR[3]))
				f.write('\n')
				f.write('DC_total_loss:\t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('DC_discriminator_crossentropy:\t'+str(loss_dev_DC[2]))
				f.write('\n')
				f.write('DC_L_diff:\t'+str(loss_dev_DC[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')

		total_epoch_count = total_epoch_count + 1

	DR_model.save_weights('DR_model_weights.h5')

	np.save('DR_development_progress.npy', np.array(DR_development_curve))
	np.save('DC_development_progress.npy', np.array(DC_development_curve))


	print('\n\n')
	print("="*400)
	print("="*400)
	print('\n\n')


	if(path.exists('DR_model_weights.h5')):
		DC_model.load_weights('DR_model_weights.h5', by_name = True)


	for epoch in range(epochs_on_a_task):
		
		DC_model.fit(X_train_D, [Y_train_D_class, Y_train_DC_task, np.ones((X_train_D.shape[0],))], batch_size = X_train_D.shape[0])
		DC_model.save_weights('DC_model_weights.h5')
		loss_dev_DC = DC_model.evaluate(X_dev_D, [Y_dev_D_class, Y_dev_DC_task, np.ones((X_dev_D.shape[0],))])

		DR_model.load_weights('DC_model_weights.h5', by_name = True)
		loss_dev_DR = DR_model.evaluate(X_dev_DR, [Y_dev_D, Y_dev_DR_task, np.ones((X_dev_D.shape[0],))])
		
		print('loss_dev_DR :', loss_dev_DR)
		print('loss_dev_DC :', loss_dev_DC)

		DR_development_curve.append([total_epoch_count] + loss_dev_DR)
		DC_development_curve.append([total_epoch_count] + loss_dev_DC)

		if(loss_dev_DR[1] < min_DR_MSE):
			print("DR_MSE "*200)

			min_DR_MSE = loss_dev_DR[1]
			DR_model.save_weights('optonDR_MSE_DR_weights.h5')
			DC_model.save_weights('optonDR_MSE_DC_weights.h5')

			with open('DR_MSE_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DR[4]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DC[4]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('DR_total_loss:\t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_discriminator_crossentropy:\t'+str(loss_dev_DR[2]))
				f.write('\n')
				f.write('DR_L_diff:\t'+str(loss_dev_DR[3]))
				f.write('\n')
				f.write('DC_total_loss:\t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('DC_discriminator_crossentropy:\t'+str(loss_dev_DC[2]))
				f.write('\n')
				f.write('DC_L_diff:\t'+str(loss_dev_DC[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')


		if(loss_dev_DC[4] > max_DC_ACC):
			print("DC_ACC "*200)

			max_DC_ACC = loss_dev_DC[4]
			DR_model.save_weights('optonDC_ACC_DR_weights.h5')
			DC_model.save_weights('optonDC_ACC_DC_weights.h5')

			with open('DC_ACC_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DR[4]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DC[4]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('DR_total_loss:\t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_discriminator_crossentropy:\t'+str(loss_dev_DR[2]))
				f.write('\n')
				f.write('DR_L_diff:\t'+str(loss_dev_DR[3]))
				f.write('\n')
				f.write('DC_total_loss:\t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('DC_discriminator_crossentropy:\t'+str(loss_dev_DC[2]))
				f.write('\n')
				f.write('DC_L_diff:\t'+str(loss_dev_DC[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')
		
		if(loss_dev_DC[1] < min_DC_crossentropy):
			print("DC_crossentropy "*20)

			min_DC_crossentropy = loss_dev_DR[1]

			DR_model.save_weights('optonDC_crossentropy_DR_weights.h5')
			DC_model.save_weights('optonDC_crossentropy_DC_weights.h5')

			with open('DC_crossentropy_current_best.txt', 'w') as f:
				f.write('DR_MSE:\t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('DR_MAE:\t'+str(loss_dev_DR[4]))
				f.write('\n')
				f.write('DC_ACC:\t'+str(loss_dev_DC[4]))
				f.write('\n')
				f.write('DC_crossentropy:\t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('DR_total_loss:\t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_discriminator_crossentropy:\t'+str(loss_dev_DR[2]))
				f.write('\n')
				f.write('DR_L_diff:\t'+str(loss_dev_DR[3]))
				f.write('\n')
				f.write('DC_total_loss:\t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('DC_discriminator_crossentropy:\t'+str(loss_dev_DC[2]))
				f.write('\n')
				f.write('DC_L_diff:\t'+str(loss_dev_DC[3]))
				f.write('\n')
				f.write('Epoch:\t'+str(total_epoch_count))
				f.write('\n')

		total_epoch_count = total_epoch_count + 1

	DC_model.save_weights('DC_model_weights.h5')

	np.save('DR_development_progress.npy', np.array(DR_development_curve))
	np.save('DC_development_progress.npy', np.array(DC_development_curve))


	print('\n\n')
	print("="*400)
	print("="*400)
	print('\n\n')