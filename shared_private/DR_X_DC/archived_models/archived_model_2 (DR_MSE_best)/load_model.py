import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout, Bidirectional, TimeDistributed, Lambda, Flatten, Activation, Multiply, Add
import keras
import keras.backend as K

def load_DR_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (400, 512,))			#	m Tx nx

	Y1 = CuDNNLSTM(180, name = 'private_DR_lstm_layer', return_sequences = True)(X)
	Y2 = CuDNNLSTM(180, name = 'shared_lstm_layer', return_sequences = True)(X)
	
	Y1 = Lambda(lambda x: K.sum(Y1, axis = 1))(Y1)
	Y2 = Lambda(lambda x: K.sum(Y2, axis = 1))(Y2)

	



	H = Concatenate(axis = -1)([Y1, Y2])

	H = Dense(65, activation = 'tanh', name = 'DR_attention_hidden_layer_1')(H)
	H = Dropout(rate = 0.25)(H)

	alpha = Dense(2, activation = 'softmax', name = 'DR_attention_output_layer')(H)




	F = Lambda(lambda x : alpha[:,0:1]*Y1 + alpha[:,1:2]*Y2, name = 'DR_attention_fusion_layer')(alpha)

	Y = Dense(90, activation = 'relu', name = 'DR_hidden_layer_1')(F)
	Y = Dropout(rate = 0.25)(Y)
	
	Y = Dense(1, activation = 'relu', name = 'DR_output_layer')(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new DAIC WOZ model.")

	return model


def load_DC_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (400, 512,))			#	m Tx nx

	Y1 = CuDNNLSTM(180, name = 'private_DC_lstm_layer', return_sequences = True)(X)
	Y2 = CuDNNLSTM(180, name = 'shared_lstm_layer', return_sequences = True)(X)
	
	Y1 = Lambda(lambda x: K.sum(Y1, axis = 1))(Y1)
	Y2 = Lambda(lambda x: K.sum(Y2, axis = 1))(Y2)

	



	H = Concatenate(axis = -1)([Y1, Y2])

	H = Dense(65, activation = 'tanh', name = 'DC_attention_hidden_layer_1')(H)
	H = Dropout(rate = 0.25)(H)

	alpha = Dense(2, activation = 'softmax', name = 'DC_attention_output_layer')(H)




	F = Lambda(lambda x : alpha[:,0:1]*Y1 + alpha[:,1:2]*Y2, name = 'DC_attention_fusion_layer')(alpha)

	Y = Dense(90, activation = 'relu', name = 'DC_hidden_layer_1')(F)
	Y = Dropout(rate = 0.25)(Y)
	
	Y = Dense(5, activation = 'softmax', name = 'DC_output_layer')(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new DAIC WOZ model.")

	return model


if __name__ == "__main__":
	m = load_CMU_MOSEI_model()