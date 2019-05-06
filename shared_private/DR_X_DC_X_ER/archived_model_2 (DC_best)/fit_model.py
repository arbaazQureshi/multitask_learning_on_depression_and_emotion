from keras.models import Model, load_model

from load_model import load_DAIC_WOZ_model, load_CMU_MOSEI_model
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


loss_funcs = {'DW_classification_output_layer' : 'categorical_crossentropy', 'DW_regression_output_layer' : 'mse'}
loss_weights = {'DW_regression_output_layer' : 1.0, 'DW_classification_output_layer' : 1.0}
metrics = {'DW_classification_output_layer' : 'accuracy', 'DW_regression_output_layer' : ['mse', 'mae']}


DW_model = load_DAIC_WOZ_model()
DW_model.compile(optimizer = 'adam', loss = loss_funcs, loss_weights = loss_weights, metrics = metrics)

CM_model = load_CMU_MOSEI_model()
CM_model.compile(optimizer = 'adam', loss = mean_squared_error)


X_train_DW, Y_train_DW, Y_train_DW_class, X_train_gender_DW = load_DAIC_WOZ_training_data()
X_dev_DW, Y_dev_DW, Y_dev_DW_class, X_dev_gender_DW = load_DAIC_WOZ_development_data()

X_train_CM, Y_train_CM = load_CMU_MOSEI_training_data()
X_dev_CM, Y_dev_CM = load_CMU_MOSEI_development_data()



mini1_R = 100000
mini1_C = 100000
mini1_C_accuracy = -1
mini2 = 100000

DW_regression_development_curve = []
DW_classification_development_curve = []
DW_total_development_curve = []
CM_development_curve = []

total_epoch_count = 0

while(True):

	if(path.exists('CM_model_weights.h5')):
		DW_model.load_weights('CM_model_weights.h5', by_name = True)

	for epoch in range(25):
		
		DW_model.fit(X_train_DW, {'DW_regression_output_layer' : np.array(Y_train_DW), 'DW_classification_output_layer' : Y_train_DW_class}, batch_size = X_train_DW.shape[0])
		DW_model.save_weights('DW_model_weights.h5')
		loss_dev_DW = DW_model.evaluate(X_dev_DW, {'DW_regression_output_layer' : np.array(Y_dev_DW), 'DW_classification_output_layer' : Y_dev_DW_class})
		
		print(loss_dev_DW)
		
		CM_model.load_weights('DW_model_weights.h5', by_name = True)
		loss_dev_CM = CM_model.evaluate(X_dev_CM, Y_dev_CM)
		
		DW_regression_development_curve.append([total_epoch_count, loss_dev_DW[1], loss_dev_DW[4]])
		DW_classification_development_curve.append([total_epoch_count, loss_dev_DW[2], loss_dev_DW[5]])
		DW_total_development_curve.append([total_epoch_count, loss_dev_DW[0]])
		CM_development_curve.append([total_epoch_count, loss_dev_CM])

		if(loss_dev_DW[1] < mini1_R):
			print("* "*1000)

			mini1_R = loss_dev_DW[1]
			DW_model.save_weights('optonDWR_DW_weights.h5')
			CM_model.save_weights('optonDWR_CM_weights.h5')

			with open('DWR_current_best.txt', 'w') as f:
				f.write(str(loss_dev_DW[1]))
				f.write('\t')
				f.write(str(loss_dev_DW[4]))
				f.write('\t')
				f.write(str(int(total_epoch_count)))


		if(loss_dev_DW[2] < mini1_C):
			print("~ "*1000)

			mini1_C = loss_dev_DW[2]
			DW_model.save_weights('optonDWC_DW_weights.h5')
			CM_model.save_weights('optonDWC_CM_weights.h5')

			with open('DWC_current_best.txt', 'w') as f:
				f.write(str(loss_dev_DW[2]))
				f.write('\t')
				f.write(str(loss_dev_DW[5]))
				f.write('\t')
				f.write(str(int(total_epoch_count)))


		if(loss_dev_DW[5] > mini1_C_accuracy):
			print("ACC "*500)

			mini1_C_accuracy = loss_dev_DW[5]
			DW_model.save_weights('optonDWC-accuracy_DW_weights.h5')
			CM_model.save_weights('optonDWC-accuracy_CM_weights.h5')

			with open('DWC-accuracy_current_best.txt', 'w') as f:
				f.write(str(loss_dev_DW[2]))
				f.write('\t')
				f.write(str(loss_dev_DW[5]))
				f.write('\t')
				f.write(str(int(total_epoch_count)))


		if(loss_dev_CM < mini2):
			print("| "*1000)

			mini2 = loss_dev_CM

			DW_model.save_weights('optonCM_DW_weights.h5')
			CM_model.save_weights('optonCM_CM_weights.h5')

			with open('CM_current_best.txt', 'w') as f:
				f.write(str(loss_dev_CM))

		total_epoch_count = total_epoch_count + 1

	DW_model.save_weights('DW_model_weights.h5')

	np.save('DW_regression_development_progress.npy', np.array(DW_regression_development_curve))
	np.save('DW_classification_development_progress.npy', np.array(DW_classification_development_curve))
	np.save('DW_total_development_curve.npy', np.array(DW_total_development_curve))
	np.save('CM_development_progress.npy', np.array(CM_development_curve))








	if(path.exists('DW_model_weights.h5')):
		CM_model.load_weights('DW_model_weights.h5', by_name = True)


	for epoch in range(25):
		
		CM_model.fit(X_train_CM, Y_train_CM, batch_size = X_train_CM.shape[0])
		CM_model.save_weights('CM_model_weights.h5')
		loss_dev_CM = CM_model.evaluate(X_dev_CM, Y_dev_CM)

		DW_model.load_weights('CM_model_weights.h5', by_name = True)
		loss_dev_DW = DW_model.evaluate(X_dev_DW, {'DW_regression_output_layer' : np.array(Y_dev_DW), 'DW_classification_output_layer' : Y_dev_DW_class})
		
		print(loss_dev_DW)
		
		DW_regression_development_curve.append([total_epoch_count, loss_dev_DW[1], loss_dev_DW[4]])
		DW_classification_development_curve.append([total_epoch_count, loss_dev_DW[2], loss_dev_DW[5]])
		DW_total_development_curve.append([total_epoch_count, loss_dev_DW[0]])
		CM_development_curve.append([total_epoch_count, loss_dev_CM])


		if(loss_dev_DW[1] < mini1_R):
			print("* "*1000)

			mini1_R = loss_dev_DW[1]
			DW_model.save_weights('optonDWR_DW_weights.h5')
			CM_model.save_weights('optonDWR_CM_weights.h5')

			with open('DWR_current_best.txt', 'w') as f:
				f.write(str(loss_dev_DW[1]))
				f.write('\t')
				f.write(str(loss_dev_DW[4]))
				f.write('\t')
				f.write(str(int(total_epoch_count)))


		if(loss_dev_DW[2] < mini1_C):
			print("~ "*1000)

			mini1_C = loss_dev_DW[2]
			DW_model.save_weights('optonDWC_DW_weights.h5')
			CM_model.save_weights('optonDWC_CM_weights.h5')

			with open('DWC_current_best.txt', 'w') as f:
				f.write(str(loss_dev_DW[2]))
				f.write('\t')
				f.write(str(loss_dev_DW[5]))
				f.write('\t')
				f.write(str(int(total_epoch_count)))


		if(loss_dev_DW[5] > mini1_C_accuracy):
			print("ACC "*500)

			mini1_C_accuracy = loss_dev_DW[5]
			DW_model.save_weights('optonDWC-accuracy_DW_weights.h5')
			CM_model.save_weights('optonDWC-accuracy_CM_weights.h5')

			with open('DWC-accuracy_current_best.txt', 'w') as f:
				f.write(str(loss_dev_DW[2]))
				f.write('\t')
				f.write(str(loss_dev_DW[5]))
				f.write('\t')
				f.write(str(int(total_epoch_count)))


		if(loss_dev_CM < mini2):
			print("| "*1000)

			mini2 = loss_dev_CM

			DW_model.save_weights('optonCM_DW_weights.h5')
			CM_model.save_weights('optonCM_CM_weights.h5')

			with open('CM_current_best.txt', 'w') as f:
				f.write(str(loss_dev_CM))

		total_epoch_count = total_epoch_count + 1

	CM_model.save_weights('CM_model_weights.h5')

	np.save('DW_regression_development_progress.npy', np.array(DW_regression_development_curve))
	np.save('DW_classification_development_progress.npy', np.array(DW_classification_development_curve))
	np.save('DW_total_development_curve.npy', np.array(DW_total_development_curve))
	np.save('CM_development_progress.npy', np.array(CM_development_curve))