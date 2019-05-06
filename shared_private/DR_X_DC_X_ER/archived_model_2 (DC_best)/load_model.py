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

	Y1 = CuDNNLSTM(180, name = 'private_DW_lstm_layer', return_sequences = True)(X)
	Y2 = CuDNNLSTM(180, name = 'shared_lstm_layer', return_sequences = True)(X)
	
	Y1 = Lambda(lambda x: K.sum(Y1, axis = 1))(Y1)
	Y2 = Lambda(lambda x: K.sum(Y2, axis = 1))(Y2)

	Y = Concatenate(axis = -1)([Y1, Y2])

	Y = Dense(90, activation = 'relu', name = 'DW_hidden_layer_1')(Y)
	
	Y = Dropout(rate = 0.25)(Y)

	YR = Dense(1, activation = None, name = 'DW_regression_output_layer')(Y)

	YC = Dense(5, activation = 'softmax', name = 'DW_classification_output_layer')(Y)

	model = Model(inputs = X, outputs = [YR, YC])

	print("Created a new DAIC WOZ model.")

	return model


def load_CMU_MOSEI_model(location=None):

	if(location != None):
		model = keras.models.load_model(location)
		print("Loaded the model.")
		return model

	X = Input(shape = (15, 512,))			#	m Tx nx
	#X_gender = Input(shape = (1,))

	Y1 = CuDNNLSTM(180, name = 'private_CM_lstm_layer', return_sequences = True)(X)
	Y2 = CuDNNLSTM(180, name = 'shared_lstm_layer', return_sequences = True)(X)

	Y = Concatenate(axis = -1)([Y1, Y2])

	Y = TimeDistributed(Dense(90, activation = 'relu', name = 'CM_hidden_layer_1'))(Y)
	Y = TimeDistributed(Dropout(rate = 0.25))(Y)
	
	Y = TimeDistributed(Dense(7, activation = None, name = 'CM_output_layer'))(Y)

	model = Model(inputs = X, outputs = Y)

	print("Created a new CMU MOSEI model.")

	return model


if __name__ == "__main__":
	m = load_CMU_MOSEI_model()