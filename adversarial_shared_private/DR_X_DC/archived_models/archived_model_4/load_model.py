import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout, Bidirectional, TimeDistributed, Lambda, Flatten, Activation, Multiply, Add
import keras
import keras.backend as K

import flipGradientTF

import os

#os.environ["CUDA_VISIBLE_DEVICES"]="4"

def load_DR_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (400, 512,))			#	m Tx nx

	Y1 = CuDNNLSTM(130, name = 'private_DR_lstm_layer', return_sequences = True)(X)
	Y2 = CuDNNLSTM(130, name = 'shared_lstm_layer', return_sequences = True)(X)
	
	Y1 = Lambda(lambda x: K.sum(Y1, axis = 1))(Y1)
	Y2 = Lambda(lambda x: K.sum(Y2, axis = 1))(Y2)

	Y_diff = Lambda(lambda x: K.mean(K.abs(K.dot(K.transpose(Y1), Y2))), name = 'DR_L_diff_layer')(Y1)

	H = Concatenate(axis = -1)([Y1, Y2])

	H = Dense(50, activation = 'tanh', name = 'DR_attention_hidden_layer_1')(H)
	H = Dropout(rate = 0.25)(H)

	alpha = Dense(2, activation = 'softmax', name = 'DR_attention_output_layer')(H)

	F = Lambda(lambda x : alpha[:,0:1]*Y1 + alpha[:,1:2]*Y2, name = 'DR_attention_fusion_layer')(alpha)

	

	YR = Dense(45, activation = 'relu', name = 'DR_hidden_layer_1')(F)
	YR = Dropout(rate = 0.25)(YR)
	
	YR = Dense(1, activation = None, name = 'DR_output_layer')(YR)


	Y_discriminator_input = flipGradientTF.GradientReversal(0.3)(Y2)

	Y_discriminator_output = Dense(40, activation = 'relu', name = 'shared_discriminator_hidden_layer_1')(Y_discriminator_input)
	Y_discriminator_output = Dropout(0.25)(Y_discriminator_output)

	Y_discriminator_output = Dense(2, activation = 'softmax', name = 'shared_discriminator_output_layer')(Y_discriminator_output)

	model = Model(inputs = X, outputs = [YR, Y_discriminator_output, Y_diff])

	print("Created a new DR model.")

	return model







def load_DC_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (400, 512,))			#	m Tx nx

	Y1 = CuDNNLSTM(130, name = 'private_DC_lstm_layer', return_sequences = True)(X)
	Y2 = CuDNNLSTM(130, name = 'shared_lstm_layer', return_sequences = True)(X)
	
	Y1 = Lambda(lambda x: K.sum(Y1, axis = 1))(Y1)
	Y2 = Lambda(lambda x: K.sum(Y2, axis = 1))(Y2)

	Y_diff = Lambda(lambda x: K.mean(K.abs(K.dot(K.transpose(Y1), Y2))), name = 'DC_L_diff_layer')(Y1)

	H = Concatenate(axis = -1)([Y1, Y2])

	H = Dense(50, activation = 'tanh', name = 'DC_attention_hidden_layer_1')(H)
	H = Dropout(rate = 0.25)(H)

	alpha = Dense(2, activation = 'softmax', name = 'DC_attention_output_layer')(H)

	F = Lambda(lambda x : alpha[:,0:1]*Y1 + alpha[:,1:2]*Y2, name = 'DC_attention_fusion_layer')(alpha)


	YC = Dense(45, activation = 'relu', name = 'DC_hidden_layer_1')(F)
	YC = Dropout(rate = 0.25)(YC)

	YC = Dense(5, activation = 'softmax', name = 'DC_output_layer')(YC)



	Y_discriminator_input = flipGradientTF.GradientReversal(0.3)(Y2)

	Y_discriminator_output = Dense(40, activation = 'relu', name = 'shared_discriminator_hidden_layer_1')(Y_discriminator_input)
	Y_discriminator_output = Dropout(0.25)(Y_discriminator_output)

	Y_discriminator_output = Dense(2, activation = 'softmax', name = 'shared_discriminator_output_layer')(Y_discriminator_output)

	model = Model(inputs = X, outputs = [YC, Y_discriminator_output, Y_diff])

	print("Created a new DC model.")

	return model


if __name__ == "__main__":
	m = load_DR_model()
	n = load_DC_model()