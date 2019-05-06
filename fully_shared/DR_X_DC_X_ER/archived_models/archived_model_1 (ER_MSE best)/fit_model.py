from keras.models import Model, load_model

from load_model import load_DR_X_DC_model, load_ER_model
from load_data import load_DAIC_WOZ_training_data, load_DAIC_WOZ_development_data, load_CMU_MOSEI_training_data, load_CMU_MOSEI_development_data
import keras
import keras.backend as K

import numpy as np
import os
from os import path

import random

os.environ["CUDA_VISIBLE_DEVICES"]="1,3"



def mean_squared_error(y_true, y_pred):
	error = K.square(y_pred - y_true)
	return K.mean(K.sum(K.sum(error, axis=2), axis = 1))

progress = []


loss_funcs = {'DR_output_layer' : 'mse', 'DC_output_layer' : 'categorical_crossentropy'}
loss_weights = {'DR_output_layer' : 1.0, 'DC_output_layer' : 1.0}
metrics = {'DR_output_layer' : ['mse', 'mae'], 'DC_output_layer' : 'accuracy'}


DR_X_DC_model = load_DR_X_DC_model()
DR_X_DC_model.compile(optimizer = 'adam', loss = loss_funcs, loss_weights = loss_weights, metrics = metrics)

ER_model = load_ER_model()
ER_model.compile(optimizer = 'adam', loss = mean_squared_error)


X_train_D, Y_train_D, Y_train_D_class, X_train_gender_D = load_DAIC_WOZ_training_data()
X_dev_D, Y_dev_D, Y_dev_D_class, X_dev_gender_D = load_DAIC_WOZ_development_data()

X_train_ER, Y_train_ER = load_CMU_MOSEI_training_data()
X_dev_ER, Y_dev_ER = load_CMU_MOSEI_development_data()



mini1_loss = 100000
mini1_accuracy = -1

mini2 = 100000

mini3 = 100000

DR_development_curve = []
DC_development_curve = []
DR_X_DC_development_curve = []
ER_development_curve = []

total_epoch_count = 0

while(total_epoch_count < 600):

	if(path.exists('ER_model_weights.h5')):
		DR_X_DC_model.load_weights('ER_model_weights.h5', by_name = True)

	for epoch in range(25):
		
		DR_X_DC_model.fit(X_train_D, {'DR_output_layer' : np.array(Y_train_D), 'DC_output_layer' : Y_train_D_class}, batch_size = X_train_D.shape[0])
		DR_X_DC_model.save_weights('DR_X_DC_model_weights.h5')
		loss_dev_DR_X_DC = DR_X_DC_model.evaluate(X_dev_D, {'DR_output_layer' : np.array(Y_dev_D), 'DC_output_layer' : Y_dev_D_class})
		
		ER_model.load_weights('DR_X_DC_model_weights.h5', by_name = True)
		loss_dev_ER = ER_model.evaluate(X_dev_ER, Y_dev_ER)
		
		DR_development_curve.append([total_epoch_count, loss_dev_DR_X_DC[1], loss_dev_DR_X_DC[4]])
		DC_development_curve.append([total_epoch_count, loss_dev_DR_X_DC[2], loss_dev_DR_X_DC[5]])
		DR_X_DC_development_curve.append([total_epoch_count, loss_dev_DR_X_DC[0]])
		ER_development_curve.append([total_epoch_count, loss_dev_ER])

		if(loss_dev_DR_X_DC[2] < mini1_loss):
			print("DC_LOSS "*200)

			mini1_loss = loss_dev_DR_X_DC[2]
			DR_X_DC_model.save_weights('optonDR_X_DC_loss_DR_X_DC_weights.h5')
			ER_model.save_weights('optonDR_X_DC_loss_ER_weights.h5')

			with open('DR_X_DC_loss_current_best.txt', 'w') as f:
				f.write('D loss: \t'+str(loss_dev_DR_X_DC[0]))
				f.write('\n')
				f.write('DR MSE: \t'+str(loss_dev_DR_X_DC[1]))
				f.write('\n')
				f.write('DR MAE: \t'+str(loss_dev_DR_X_DC[4]))
				f.write('\n')
				f.write('categorical_crossentropy: \t'+str(loss_dev_DR_X_DC[2]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DR_X_DC[5]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))


		if(loss_dev_DR_X_DC[5] > mini1_accuracy):
			print("DC_ACC "*200)

			mini1_accuracy = loss_dev_DR_X_DC[5]
			DR_X_DC_model.save_weights('optonDR_X_DC_accuracy_DR_X_DC_weights.h5')
			ER_model.save_weights('optonDR_X_DC_accuracy_ER_weights.h5')

			with open('DR_X_DC_accuracy_current_best.txt', 'w') as f:
				f.write('D loss: \t'+str(loss_dev_DR_X_DC[0]))
				f.write('\n')
				f.write('DR MSE: \t'+str(loss_dev_DR_X_DC[1]))
				f.write('\n')
				f.write('DR MAE: \t'+str(loss_dev_DR_X_DC[4]))
				f.write('\n')
				f.write('categorical_crossentropy: \t'+str(loss_dev_DR_X_DC[2]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DR_X_DC[5]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))


		if(loss_dev_DR_X_DC[1] < mini2):
			print("DR_MSE "*200)

			mini2 = loss_dev_DR_X_DC[1]

			DR_X_DC_model.save_weights('optonDR_X_DC_MSE_DR_X_DC_weights.h5')
			ER_model.save_weights('optonDR_X_DC_MSE_ER_weights.h5')

			with open('DR_X_DC_MSE_current_best.txt', 'w') as f:
				f.write('D loss: \t'+str(loss_dev_DR_X_DC[0]))
				f.write('\n')
				f.write('DR MSE: \t'+str(loss_dev_DR_X_DC[1]))
				f.write('\n')
				f.write('DR MAE: \t'+str(loss_dev_DR_X_DC[4]))
				f.write('\n')
				f.write('categorical_crossentropy: \t'+str(loss_dev_DR_X_DC[2]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DR_X_DC[5]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))



		if(loss_dev_ER < mini2):
			print("ER_MSE "*100)

			mini2 = loss_dev_ER

			DR_X_DC_model.save_weights('optonER_DR_X_DC_weights.h5')
			ER_model.save_weights('optonER_ER_weights.h5')

			with open('ER_current_best.txt', 'w') as f:
				f.write('D loss: \t'+str(loss_dev_DR_X_DC[0]))
				f.write('\n')
				f.write('DR MSE: \t'+str(loss_dev_DR_X_DC[1]))
				f.write('\n')
				f.write('DR MAE: \t'+str(loss_dev_DR_X_DC[4]))
				f.write('\n')
				f.write('categorical_crossentropy: \t'+str(loss_dev_DR_X_DC[2]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DR_X_DC[5]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))

		total_epoch_count = total_epoch_count + 1

	DR_X_DC_model.save_weights('DR_X_DC_model_weights.h5')











	if(path.exists('DR_X_DC_model_weights.h5')):
		ER_model.load_weights('DR_X_DC_model_weights.h5', by_name = True)


	for epoch in range(25):
		
		ER_model.fit(X_train_ER, Y_train_ER, batch_size = X_train_ER.shape[0])
		ER_model.save_weights('ER_model_weights.h5')
		loss_dev_ER = ER_model.evaluate(X_dev_ER, Y_dev_ER)

		DR_X_DC_model.load_weights('ER_model_weights.h5', by_name = True)
		loss_dev_DR_X_DC = DR_X_DC_model.evaluate(X_dev_D, {'DR_output_layer' : np.array(Y_dev_D), 'DC_output_layer' : Y_dev_D_class})
		
		DR_development_curve.append([total_epoch_count, loss_dev_DR_X_DC[1], loss_dev_DR_X_DC[4]])
		DC_development_curve.append([total_epoch_count, loss_dev_DR_X_DC[2], loss_dev_DR_X_DC[5]])
		DR_X_DC_development_curve.append([total_epoch_count, loss_dev_DR_X_DC[0]])
		ER_development_curve.append([total_epoch_count, loss_dev_ER])


		if(loss_dev_DR_X_DC[2] < mini1_loss):
			print("DC_LOSS "*200)

			mini1_loss = loss_dev_DR_X_DC[2]
			DR_X_DC_model.save_weights('optonDR_X_DC_loss_DR_X_DC_weights.h5')
			ER_model.save_weights('optonDR_X_DC_loss_ER_weights.h5')

			with open('DR_X_DC_loss_current_best.txt', 'w') as f:
				f.write('D loss: \t'+str(loss_dev_DR_X_DC[0]))
				f.write('\n')
				f.write('DR MSE: \t'+str(loss_dev_DR_X_DC[1]))
				f.write('\n')
				f.write('DR MAE: \t'+str(loss_dev_DR_X_DC[4]))
				f.write('\n')
				f.write('categorical_crossentropy: \t'+str(loss_dev_DR_X_DC[2]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DR_X_DC[5]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))


		if(loss_dev_DR_X_DC[5] > mini1_accuracy):
			print("DC_ACC "*200)

			mini1_accuracy = loss_dev_DR_X_DC[5]
			DR_X_DC_model.save_weights('optonDR_X_DC_accuracy_DR_X_DC_weights.h5')
			ER_model.save_weights('optonDR_X_DC_accuracy_ER_weights.h5')

			with open('DR_X_DC_accuracy_current_best.txt', 'w') as f:
				f.write('D loss: \t'+str(loss_dev_DR_X_DC[0]))
				f.write('\n')
				f.write('DR MSE: \t'+str(loss_dev_DR_X_DC[1]))
				f.write('\n')
				f.write('DR MAE: \t'+str(loss_dev_DR_X_DC[4]))
				f.write('\n')
				f.write('categorical_crossentropy: \t'+str(loss_dev_DR_X_DC[2]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DR_X_DC[5]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))


		if(loss_dev_DR_X_DC[1] < mini2):
			print("DR_MSE "*200)

			mini2 = loss_dev_DR_X_DC[1]

			DR_X_DC_model.save_weights('optonDR_X_DC_MSE_DR_X_DC_weights.h5')
			ER_model.save_weights('optonDR_X_DC_MSE_ER_weights.h5')

			with open('DR_X_DC_MSE_current_best.txt', 'w') as f:
				f.write('D loss: \t'+str(loss_dev_DR_X_DC[0]))
				f.write('\n')
				f.write('DR MSE: \t'+str(loss_dev_DR_X_DC[1]))
				f.write('\n')
				f.write('DR MAE: \t'+str(loss_dev_DR_X_DC[4]))
				f.write('\n')
				f.write('categorical_crossentropy: \t'+str(loss_dev_DR_X_DC[2]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DR_X_DC[5]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))



		if(loss_dev_ER < mini2):
			print("ER_MSE "*100)

			mini2 = loss_dev_ER

			DR_X_DC_model.save_weights('optonER_DR_X_DC_weights.h5')
			ER_model.save_weights('optonER_ER_weights.h5')

			with open('ER_current_best.txt', 'w') as f:
				f.write('D loss: \t'+str(loss_dev_DR_X_DC[0]))
				f.write('\n')
				f.write('DR MSE: \t'+str(loss_dev_DR_X_DC[1]))
				f.write('\n')
				f.write('DR MAE: \t'+str(loss_dev_DR_X_DC[4]))
				f.write('\n')
				f.write('categorical_crossentropy: \t'+str(loss_dev_DR_X_DC[2]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DR_X_DC[5]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))


		total_epoch_count = total_epoch_count + 1

	ER_model.save_weights('ER_model_weights.h5')

	np.save('DR_development_curve', DR_development_curve.append([total_epoch_count, loss_dev_DR_X_DC[1], loss_dev_DR_X_DC[4]]))
	np.save('DC_development_curve', DC_development_curve.append([total_epoch_count, loss_dev_DR_X_DC[2], loss_dev_DR_X_DC[5]]))
	np.save('DR_X_DC_development_curve', DR_X_DC_development_curve.append([total_epoch_count, loss_dev_DR_X_DC[0]]))
	np.save('ER_development_curve', ER_development_curve.append([total_epoch_count, loss_dev_ER]))