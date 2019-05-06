from keras.models import Model, load_model

from load_model import load_model
from load_data import load_DAIC_WOZ_training_data as load_training_data
from load_data import load_DAIC_WOZ_development_data as load_development_data
import keras

import numpy as np
import os
from os import path

import random

os.environ["CUDA_VISIBLE_DEVICES"]="6"

progress = []

model = load_model()
model.compile(optimizer='adam', loss='mse', metrics = ['mae'])

X_train, Y_train, X_train_gender = load_training_data()
X_dev, Y_dev, X_dev_gender = load_development_data()

min_loss_dev = 10000
min_mae = 10000
loss_dev = [10000, 10000]

current_epoch_count = 1
total_number_of_epochs = 200

#batch_size_list = list(range(1, 25))
#no_of_downward_epochs = 1000

m = X_train.shape[0]

print("\n\n")



while(current_epoch_count < total_number_of_epochs):
	
	print((str(current_epoch_count) + ' ')*30)
	print(total_number_of_epochs - current_epoch_count, "epochs to go.")

	#prev_loss_dev = loss_dev[0]
	batch_size = m
	
	#print("Batch size is", batch_size)
	
	hist = model.fit(X_train, Y_train, batch_size = batch_size, epochs = 1)

	loss_train = hist.history['loss'][-1]
	loss_dev = model.evaluate(X_dev, Y_dev, batch_size = int(X_dev.shape[0]/2))

	print(loss_train, loss_dev[0], loss_dev[1])

	if(loss_dev[0] < min_loss_dev):
		min_loss_dev = loss_dev[0]
		min_mae = loss_dev[1]

		print("SAVING THE WEIGHTS!\n\n")
		model.save_weights('optimal_weights.h5')
		
		np.savetxt('BEST_TILL_NOW.txt', np.array([min_loss_dev, min_mae, current_epoch_count]), fmt='%.4f')

	progress.append([current_epoch_count, loss_train, loss_dev[0], loss_dev[1]])
	np.savetxt('training_progress.csv', np.array(progress), fmt='%.4f', delimiter=',')
	

	current_epoch_count = current_epoch_count + 1
	print("\n\n")