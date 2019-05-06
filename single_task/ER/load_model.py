import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, Input, Concatenate, Dropout, Bidirectional, TimeDistributed, Lambda, Flatten, Activation, Multiply, Add
import keras
import keras.backend as K

def load_DAIC_WOZ_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (400, 512,))			#	m Tx nx
	X_gender = Input(shape = (1,))

	Y = CuDNNLSTM(180, name = 'DW_specific_lstm_layer', return_sequences = True)(X)
	
	Y = Lambda(lambda x: K.sum(Y, axis = 1))(Y)
	Y = Dropout(rate = 0.25)(Y)

	Y = Dense(70, activation = 'relu', name = 'regressor_hidden_layer')(Y)
	Y = Dropout(rate = 0.25)(Y)
	
	Y = Dense(1, activation = None, name = 'regressor_output_layer')(Y)

	model = Model(inputs = [X, X_gender], outputs = Y)

	print("Created a new model.")

	return model


def load_CMU_MOSEI_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (15, 512,))			#	m Tx nx

	Y = CuDNNLSTM(180, name = 'CM_specific_lstm_layer', return_sequences = True)(X)
	Y = TimeDistributed(Dropout(rate = 0.25))(Y)

	Y = TimeDistributed(Dense(70, activation = 'relu', name = 'CM_hidden_layer_1'))(Y)
	Y = TimeDistributed(Dropout(rate = 0.25))(Y)
	
	Y = TimeDistributed(Dense(7, activation = None, name = 'CM_output_layer'))(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new model.")

	return model


if __name__ == "__main__":
	m = load_CMU_MOSEI_model()