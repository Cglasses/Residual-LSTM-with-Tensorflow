import tensorflow as tf
from itertools import chain
from simple_tensor.tensor_operations import * 
import numpy as np


class LSTM(object):
	def __init__(self, input_feature_num, output_feature_num, memory_feature_num, dropout_val=0.75):
		"""
		CLSTM Constructor
		Args:
			input_feature_num   :		an integer, the number of input feature
			output_feature_num  :		an integer, the number of the output feture
			memory_feature_num  :       an integer, the number of LSTM feature memory
		"""
		# the number of feature as input vector
		self.input_feature_num = input_feature_num
		# the number of output feature
		self.output_feature_num = output_feature_num
		# the number of memory feature
		self.memory_feature_num = memory_feature_num
		# the number of feture feed to neural net block inside LSTM
		self.nn_inside_LSTM_inputfeature_num = (self.input_feature_num + self.output_feature_num)
		# the number of result feature of neural net1 block inside LSTM
		self.nn1_inside_LSTM_outputfeature_num = memory_feature_num
		# the number of result feature of neural net2 block inside LSTM
		self.nn2_inside_LSTM_outputfeature_num = memory_feature_num
		# the number of result feature of neural net3 block inside LSTM
		self.nn3_inside_LSTM_outputfeature_num = memory_feature_num
		# the number of result feature of neural net4 block inside LSTM
		self.nn4_inside_LSTM_outputfeature_num = memory_feature_num

		# dropout presentation
		self.dropout_val = dropout_val

		# placeholder for first neural net block
		self.inside_LSTM_nn_input = tf.placeholder(tf.float32, shape=(None, self.nn_inside_LSTM_inputfeature_num))

	
	def inside_LSTM_nn(self, layer_out_num1, layer_out_num2, layer_out_num3, nn_code, cell_code):
		"""
		A function of neural netwok block inside LSTM. 
		Args:
			layer_out_num1      :		an integer, the number of output from layer 1 int this block
			layer_out_num2      :		an integer, the number of output from layer 2 int this block
			layer_out_num3      :		an integer, the number of output from layer 3 int this block
			nn_code             :       a string, the code for this block (just for graph naming)
			cell_code           :       a string, the code for the LSTM cell (just for graph naming)
		Return:
			the output tensor and variable list of this block
		"""
		# first convolution layer
		fc1, w_fc1, b_fc1 = new_fc_layer(self.inside_LSTM_nn_input, self.nn_inside_LSTM_inputfeature_num, layer_out_num1, \
											name='fc1_nn' + nn_code +"_" + cell_code, activation="LRELU")
		#bn1, bn1_b, bn1_g = batch_norm(fc1, 1, name='bn1' + nn_code +"_" + cell_code, is_convolution=False)
		drop1 = tf.nn.dropout(fc1, self.dropout_val)
		# second convolution layer
		fc2, w_fc2, b_fc2 = new_fc_layer(drop1, layer_out_num1, layer_out_num2,\
											name='fc2_nn' + nn_code +"_" + cell_code, activation="LRELU")
		#bn2, bn2_b, bn2_g = batch_norm(fc2, 1, name='bn2' + nn_code +"_" + cell_code, is_convolution=False)
		drop2 = tf.nn.dropout(fc2, self.dropout_val)
		# third convolution layer
		fc3, w_fc3, b_fc3 = new_fc_layer(drop2, layer_out_num2, layer_out_num3,\
											 name='fc3_nn' + nn_code +"_" + cell_code, activation="none")
		drop3 = tf.nn.dropout(fc3, self.dropout_val)
		vars = [w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3] #, bn1_b, bn1_g, bn2_b, bn2_g]
		return drop3, vars


	def build_lstm_cell(self, last_output, last_memmory, input_tensor, num_hidden_neuron=10, cell_code='1'):
		"""
		A function of LSTM cell#self.input_feature_num = input_feature_num
		#self.output_feature_num = output_feature_num
		#self.memory_feature_num = memory_feature_num
		Args:
			last_output         :       A tensor or numpy, the output from previous lstm cell
			last_memmory        :       A tensor or numpy, the memory from previous lstm cell
			input_tensor        :       A tensor or numpy, the input feature
			cell_code           :       a string, the code for the LSTM cell (just for graph naming)
		Return:
			the output tensor, feature memory and variable list of this LSTM cell
		"""
		s1 = tf.concat([input_tensor, last_output], axis=1, name='concat_' + str(cell_code))
		self.inside_LSTM_nn_input = s1

		s2, s2_vars = self.inside_LSTM_nn(num_hidden_neuron, num_hidden_neuron, self.nn1_inside_LSTM_outputfeature_num, '1', cell_code)
		s2 = tf.nn.sigmoid(s2, name = 's2_sig_' + cell_code)

		s3, s3_vars = self.inside_LSTM_nn(num_hidden_neuron, num_hidden_neuron, self.nn2_inside_LSTM_outputfeature_num, '2', cell_code)
		s3 = tf.nn.sigmoid(s3, name = 's3_sig_' + cell_code)

		s4, s4_vars = self.inside_LSTM_nn(num_hidden_neuron, num_hidden_neuron, self.nn3_inside_LSTM_outputfeature_num, '3', cell_code)
		s4 = tf.tanh(s4, name='tanh_s4_' + cell_code)

		s5, s5_vars = self.inside_LSTM_nn(num_hidden_neuron, num_hidden_neuron, self.nn4_inside_LSTM_outputfeature_num, '4', cell_code)
		s5 = tf.nn.sigmoid(s5, name = 's5_sig_' + cell_code)

		s6 = tf.multiply(s3, s4, name = 'multiply_s3_s4_' + cell_code)
		#bns6, bns6_b, bns6_g = batch_norm(s6, 1, name='bns6' + "_" + cell_code, is_convolution=False)

		s7 = tf.multiply(last_memmory, s2, name = 'multiply_s2_memory_' + cell_code)
		#bns7, bns7_b, bns7_g = batch_norm(s7, 1, name='bns7' + "_" + cell_code, is_convolution=False)

		s8 = tf.add(s6, s7, name='add_s6_s7_' + cell_code)

		s9 = tf.tanh(s8, name='tanh_s8_' + cell_code)

		s10 = tf.multiply(s5, s9, name = 'multiply_s5_s10_' + cell_code)
		drop_s10 = tf.nn.dropout(s10, self.dropout_val)

		pre_out, w_out, b_out = new_fc_layer(drop_s10, self.memory_feature_num, self.output_feature_num,\
											 name='out_nn_' + cell_code, activation="none")
		#out = tf.nn.sigmoid(pre_out)

		#bns6_vars = [bns6_b, bns6_g]
		#bns7_vars = [bns7_b, bns7_g]
		s10_vars = [w_out, b_out]
		LSTM_vars =  list(chain(*[s2_vars, s3_vars, s4_vars, s5_vars, s10_vars])) #, bns6_vars, bns7_vars]))
		return pre_out, s8, LSTM_vars

	def mse_loss(self, predicted, evaluate_all_output=True, slice_from_last = -1):
		"""
		A function for calculating the loss
		Args:
			predicted         :       A tensor, the prediction result
		Return:
			a single value of integer tensor 
		"""
		if evaluate_all_output:
			actual = self.output
		else:
			actual = self.output[:, slice_from_last:, :]
		loss = tf.subtract(predicted, actual)
		loss = tf.square(loss)
		loss = tf.reduce_mean(loss)
		return loss

	def mape_loss(self, predicted, evaluate_all_output=True, decay=0.0001):
		"""
		A function for calculating the loss
		Args:
			predicted         :       A tensor, the prediction result
		Return:
			a single value of integer tensor 
		"""
		loss = tf.abs(tf.subtract(predicted, self.output))/( self.output + decay)
		loss = tf.reduce_mean(loss)
		return loss



class GRU():
	"""Implementation of a Gated Recurrent Unit (GRU) as described in [1].
	
	[1] Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.
	
	Arguments
	---------
	input_dimensions: int
		The size of the input vectors (x_t).
	hidden_size: int
		The size of the hidden layer vectors (h_t).
	dtype: obj
		The datatype used for the variables and constants (optional).
	"""
	
	def __init__(self, input_feature_num, output_feature_num, memory_feature_num, dropout_val=0.75):
		# the number of feature as input vector
		self.input_feature_num = input_feature_num
		# the number of output feature
		self.output_feature_num = output_feature_num
		# the number of memory feature
		self.memory_feature_num = memory_feature_num
		# the number of feture feed to neural net block inside LSTM
		self.nn_inside_LSTM_inputfeature_num = (self.input_feature_num + self.output_feature_num)
		# the number of result feature of neural net1 block inside LSTM
		self.nn1_inside_LSTM_outputfeature_num = memory_feature_num
		# the number of result feature of neural net2 block inside LSTM
		self.nn2_inside_LSTM_outputfeature_num = memory_feature_num
		# the number of result feature of neural net3 block inside LSTM
		self.nn3_inside_LSTM_outputfeature_num = memory_feature_num
		# the number of result feature of neural net4 block inside LSTM
		self.nn4_inside_LSTM_outputfeature_num = memory_feature_num

	def build_gru_cell(self, h_tm1, x_t):
		"""Perform a forward pass.
		Arguments
		---------
		h_tm1: np.matrix
			The hidden state at the previous timestep (h_{t-1}).
		x_t: np.matrix
			The input vector.
		"""
		# Definitions of z_t and r_t
		z_t = 0



class SimpleSquenceLSTM(LSTM):
	def __init__(self, batch_size, num_lstm_cell, input_feature_num, output_feature_num, memory_feature_num, 
					hidden_neuron_num=10, dropout_val=0.95, with_residual= False, residual_season=1, return_memmory=False):
		"""
		A Constructor
		Args:
			batch_size          :       an integer, the size of batch
			num_lstm_cell       :       an integer, the number of lstm cell
			input_feature_num   :       an integer, the number of input feature
			output_feature_num  :       an integer, the number of output feature
			memory_feature_num  :       an integer, the number of feature in memory
			residual_season     :       an integer, ss
		"""
		super(SimpleSquenceLSTM, self).__init__(input_feature_num, output_feature_num, memory_feature_num, dropout_val=0.95)
		self.batch_size = batch_size
		self.num_lstm_cell = num_lstm_cell
		self.num_hidden_neuron = hidden_neuron_num
		self.residual_season = residual_season
		self.output_shape = [batch_size, num_lstm_cell, output_feature_num]
		self.with_residual = with_residual
		self.return_memmory = return_memmory

		# gate for input features
		self.input_feature_placeholder = tf.placeholder(tf.float32, shape=(None, num_lstm_cell, self.input_feature_num), name='input_feature_placeholder')
		# gate for target
		self.output = tf.placeholder(tf.float32, shape=(None, num_lstm_cell, self.output_feature_num), name='output_placeholder')
	
	def build_net(self, scoope=''):
		"""
		A function for buliding sequence of LSTM
		Return:
			the output tensor and variable list of this LSTM cell
		"""
		outs = []
		cell_vars = []
		memmories = {}
		for i in range(self.num_lstm_cell):
			if i == 0:
				last_output = tf.convert_to_tensor(np.zeros((self.batch_size, self.output_feature_num)).astype(np.float32))
				last_memmory = tf.convert_to_tensor(np.zeros((self.batch_size, self.memory_feature_num)).astype(np.float32))

			cell_input = tf.reshape(self.input_feature_placeholder[:, i, :], [self.batch_size, self.input_feature_num])
			out, memory, cell_var = self.build_lstm_cell(last_output, last_memmory, cell_input, num_hidden_neuron = self.num_hidden_neuron, cell_code= scoope + str(i))
			last_output = out
			memmories[i] = memory
			if self.with_residual and i >= self.residual_season:
				last_memmory = 0.9 * memory + 0.1 * memmories[i-self.residual_season]
			else:
				last_memmory = memory 
			outs.append(out)
			cell_vars.extend(cell_var)

		outs = tf.reshape(outs, [self.batch_size, self.num_lstm_cell, self.output_feature_num])

		if self.return_memmory:
			memmories = tf.reshape(list(memmories.values()), [self.batch_size, self.num_lstm_cell, self.memory_feature_num])
			return outs, memmories, cell_vars
		else:
			return outs, cell_vars
	
	





		