from keras.models import Model, load_model

from load_model import load_DC_model, load_ER_model
from load_data import load_DAIC_WOZ_training_data, load_DAIC_WOZ_development_data, load_CMU_MOSEI_training_data, load_CMU_MOSEI_development_data
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


DC_model = load_DC_model()
DC_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

ER_model = load_ER_model()
ER_model.compile(optimizer = 'adam', loss = mean_squared_error)


X_train_DC, Y_train_DC, X_train_gender_DC = load_DAIC_WOZ_training_data()

#X_train_DC = np.array(X_train_DC)
#X_train_gender_DC = np.array(X_train_gender_DC)

X_dev_DC, Y_dev_DC, X_dev_gender_DC = load_DAIC_WOZ_development_data()

#X_dev_DC = np.array(X_dev_DC)
#X_dev_gender_DC = np.array(X_dev_gender_DC)


X_train_ER, Y_train_ER = load_CMU_MOSEI_training_data()
#X_train_ER = np.array(X_train_ER)

X_dev_ER, Y_dev_ER = load_CMU_MOSEI_development_data()
#X_dev_ER = np.array(X_dev_ER)



mini1_loss = 100000
mini1_accuracy = -1
mini2 = 100000

DC_development_curve = []
ER_development_curve = []

total_epoch_count = 0

while(total_epoch_count < 800):

	if(path.exists('ER_model_weights.h5')):
		DC_model.load_weights('ER_model_weights.h5', by_name = True)

	for epoch in range(25):
		
		DC_model.fit(X_train_DC, Y_train_DC, batch_size = X_train_DC.shape[0])
		DC_model.save_weights('DC_model_weights.h5')
		loss_dev_DC = DC_model.evaluate(X_dev_DC, Y_dev_DC)
		
		ER_model.load_weights('DC_model_weights.h5', by_name = True)
		loss_dev_ER = ER_model.evaluate(X_dev_ER, Y_dev_ER)
		
		DC_development_curve.append([total_epoch_count, loss_dev_DC[0], loss_dev_DC[1]])
		ER_development_curve.append([total_epoch_count, loss_dev_ER])

		if(loss_dev_DC[0] < mini1_loss):
			print("* "*500)

			mini1_loss = loss_dev_DC[0]
			DC_model.save_weights('optonDC_loss_DC_weights.h5')
			ER_model.save_weights('optonDC_loss_ER_weights.h5')

			with open('DC_loss_current_best.txt', 'w') as f:
				f.write('categorical_crossentropy: \t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))


		if(loss_dev_DC[1] > mini1_accuracy):
			print("~ "*500)

			mini1_accuracy = loss_dev_DC[1]
			DC_model.save_weights('optonDC_accuracy_DC_weights.h5')
			ER_model.save_weights('optonDC_accuracy_ER_weights.h5')

			with open('DC_accuracy_current_best.txt', 'w') as f:
				f.write('categorical_crossentropy: \t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))


		if(loss_dev_ER < mini2):
			print("| "*500)

			mini2 = loss_dev_ER

			DC_model.save_weights('optonER_DC_weights.h5')
			ER_model.save_weights('optonER_ER_weights.h5')

			with open('ER_current_best.txt', 'w') as f:
				f.write('categorical_crossentropy: \t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))

		total_epoch_count = total_epoch_count + 1

	DC_model.save_weights('DC_model_weights.h5')











	if(path.exists('DC_model_weights.h5')):
		ER_model.load_weights('DC_model_weights.h5', by_name = True)


	for epoch in range(25):
		
		ER_model.fit(X_train_ER, Y_train_ER, batch_size = X_train_ER.shape[0])
		ER_model.save_weights('ER_model_weights.h5')
		loss_dev_ER = ER_model.evaluate(X_dev_ER, Y_dev_ER)

		DC_model.load_weights('ER_model_weights.h5', by_name = True)
		loss_dev_DC = DC_model.evaluate(X_dev_DC, Y_dev_DC)
		
		DC_development_curve.append([total_epoch_count, loss_dev_DC[0], loss_dev_DC[1]])
		ER_development_curve.append([total_epoch_count, loss_dev_ER])


		if(loss_dev_DC[0] < mini1_loss):
			print("* "*500)

			mini1_loss = loss_dev_DC[0]
			DC_model.save_weights('optonDC_loss_DC_weights.h5')
			ER_model.save_weights('optonDC_loss_ER_weights.h5')

			with open('DC_loss_current_best.txt', 'w') as f:
				f.write('categorical_crossentropy: \t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))


		if(loss_dev_DC[1] > mini1_accuracy):
			print("~ "*500)

			mini1_accuracy = loss_dev_DC[1]
			DC_model.save_weights('optonDC_accuracy_DC_weights.h5')
			ER_model.save_weights('optonDC_accuracy_ER_weights.h5')

			with open('DC_accuracy_current_best.txt', 'w') as f:
				f.write('categorical_crossentropy: \t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))

		if(loss_dev_ER < mini2):
			print("| "*500)

			mini2 = loss_dev_ER

			DC_model.save_weights('optonER_DC_weights.h5')
			ER_model.save_weights('optonER_ER_weights.h5')

			with open('ER_current_best.txt', 'w') as f:
				f.write('categorical_crossentropy: \t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('ER_MSE: \t'+str(loss_dev_ER))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))

		total_epoch_count = total_epoch_count + 1

	ER_model.save_weights('ER_model_weights.h5')

	np.save('DC_development_progress.npy', np.array(DC_development_curve))
	np.save('ER_development_progress.npy', np.array(ER_development_curve))