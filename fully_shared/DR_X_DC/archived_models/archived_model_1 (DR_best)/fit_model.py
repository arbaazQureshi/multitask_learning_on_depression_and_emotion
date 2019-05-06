from keras.models import Model, load_model

from load_model import load_DC_model, load_DR_model
from load_data import load_DAIC_WOZ_training_data, load_DAIC_WOZ_development_data, load_CMU_MOSEI_training_data, load_CMU_MOSEI_development_data
import keras
import keras.backend as K

import numpy as np
import os
from os import path

import random

os.environ["CUDA_VISIBLE_DEVICES"]="1"

DC_model = load_DC_model()
DC_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

DR_model = load_DR_model()
DR_model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])


X_train_DC, Y_train_DC, Y_train_DC_class, X_train_gendDR_DC = load_DAIC_WOZ_training_data()

X_dev_DC, Y_dev_DC, Y_dev_DC_class, X_dev_gendDR_DC = load_DAIC_WOZ_development_data()

mini1_loss = 100000
mini1_accuracy = -1

mini2 = 100000

DC_development_curve = []
DR_development_curve = []

total_epoch_count = 0

while(total_epoch_count < 500):

	if(path.exists('DR_model_weights.h5')):
		DC_model.load_weights('DR_model_weights.h5', by_name = True)

	for epoch in range(25):
		
		DC_model.fit(X_train_DC, Y_train_DC_class, batch_size = X_train_DC.shape[0])
		DC_model.save_weights('DC_model_weights.h5')
		loss_dev_DC = DC_model.evaluate(X_dev_DC, Y_dev_DC_class)
		
		DR_model.load_weights('DC_model_weights.h5', by_name = True)
		loss_dev_DR = DR_model.evaluate(X_dev_DC, Y_dev_DC)
		
		DC_development_curve.append([total_epoch_count, loss_dev_DC[0], loss_dev_DC[1]])
		DR_development_curve.append([total_epoch_count, loss_dev_DR[0], loss_dev_DR[1]])

		if(loss_dev_DC[0] < mini1_loss):
			print("* "*500)

			mini1_loss = loss_dev_DC[0]
			DC_model.save_weights('optonDC_loss_DC_weights.h5')
			DR_model.save_weights('optonDC_loss_DR_weights.h5')

			with open('DC_loss_current_best.txt', 'w') as f:
				f.write('categorical_crossentropy: \t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('DR_MSE: \t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_MAE: \t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))


		if(loss_dev_DC[1] > mini1_accuracy):
			print("~ "*500)

			mini1_accuracy = loss_dev_DC[1]
			DC_model.save_weights('optonDC_accuracy_DC_weights.h5')
			DR_model.save_weights('optonDC_accuracy_DR_weights.h5')

			with open('DC_accuracy_current_best.txt', 'w') as f:
				f.write('categorical_crossentropy: \t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('DR_MSE: \t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_MAE: \t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))


		if(loss_dev_DR[0] < mini2):
			print("| "*500)

			mini2 = loss_dev_DR[0]

			DC_model.save_weights('optonDR_DC_weights.h5')
			DR_model.save_weights('optonDR_DR_weights.h5')

			with open('DR_current_best.txt', 'w') as f:
				f.write('categorical_crossentropy: \t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('DR_MSE: \t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_MAE: \t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))

		total_epoch_count = total_epoch_count + 1

	DC_model.save_weights('DC_model_weights.h5')











	if(path.exists('DC_model_weights.h5')):
		DR_model.load_weights('DC_model_weights.h5', by_name = True)


	for epoch in range(25):
		
		DR_model.fit(X_train_DC, Y_train_DC, batch_size = X_train_DC.shape[0])
		DR_model.save_weights('DR_model_weights.h5')
		loss_dev_DR = DR_model.evaluate(X_dev_DC, Y_dev_DC)

		DC_model.load_weights('DR_model_weights.h5', by_name = True)
		loss_dev_DC = DC_model.evaluate(X_dev_DC, Y_dev_DC_class)
		
		DC_development_curve.append([total_epoch_count, loss_dev_DC[0], loss_dev_DC[1]])
		DR_development_curve.append([total_epoch_count, loss_dev_DR[0], loss_dev_DR[1]])


		if(loss_dev_DC[0] < mini1_loss):
			print("* "*500)

			mini1_loss = loss_dev_DC[0]
			DC_model.save_weights('optonDC_loss_DC_weights.h5')
			DR_model.save_weights('optonDC_loss_DR_weights.h5')

			with open('DC_loss_current_best.txt', 'w') as f:
				f.write('categorical_crossentropy: \t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('DR_MSE: \t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_MAE: \t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))


		if(loss_dev_DC[1] > mini1_accuracy):
			print("~ "*500)

			mini1_accuracy = loss_dev_DC[1]
			DC_model.save_weights('optonDC_accuracy_DC_weights.h5')
			DR_model.save_weights('optonDC_accuracy_DR_weights.h5')

			with open('DC_accuracy_current_best.txt', 'w') as f:
				f.write('categorical_crossentropy: \t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('DR_MSE: \t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_MAE: \t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))

		if(loss_dev_DR[0] < mini2):
			print("| "*500)

			mini2 = loss_dev_DR[0]

			DC_model.save_weights('optonDR_DC_weights.h5')
			DR_model.save_weights('optonDR_DR_weights.h5')

			with open('DR_current_best.txt', 'w') as f:
				f.write('categorical_crossentropy: \t'+str(loss_dev_DC[0]))
				f.write('\n')
				f.write('accuracy: \t'+str(loss_dev_DC[1]))
				f.write('\n')
				f.write('DR_MSE: \t'+str(loss_dev_DR[0]))
				f.write('\n')
				f.write('DR_MAE: \t'+str(loss_dev_DR[1]))
				f.write('\n')
				f.write('epoch: \t'+str(total_epoch_count))

		total_epoch_count = total_epoch_count + 1

	DR_model.save_weights('DR_model_weights.h5')

	np.save('DC_development_progress.npy', np.array(DC_development_curve))
	np.save('DR_development_progress.npy', np.array(DR_development_curve))