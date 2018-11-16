import tensorflow as tf
from itertools import chain
from simple_tensor.tensor_operations import * 


class LSTM():
    def __init__(self, input_feature_num, output_feature_num, memory_feature_num):
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

        # gate for input features
        self.input_feature_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_feature_num), name='input_feature_placeholder')
        # gate for otput feature from previous LSTM
        self.output_feature_placeholder = tf.placeholder(tf.float32, shape=(None, self.output_feature_num), name='output_feature_placeholder')
        # gate for memory from previous LSTM cell
        self.memory_feature_placeholder = tf.placeholder(tf.float32, shape=(None, self.memory_feature_num), name='memory_feature_placeholder')

        # placeholder for first neural net block
        self.inside_LSTM_nn_placeholder = tf.placeholder(tf.float32, shape=(None, self.nn_inside_LSTM_feature_num))

    
    def inside_LSTM_nn(self, layer_out_num1, layer_out_num2, layer_out_num3, layer_out_num4, layer_out_num5, nn_code, cell_code):
        """
        A function of neural netwok block inside LSTM. 
        Args:
            layer_out_num1      :		an integer, the number of output from layer 1 int this block
            layer_out_num2      :		an integer, the number of output from layer 2 int this block
            layer_out_num3      :		an integer, the number of output from layer 3 int this block
            layer_out_num4      :		an integer, the number of output from layer 4 int this block
            layer_out_num5      :		an integer, the number of output from layer 5 int this block
            nn_code             :       a string, the code for this block (just for graph naming)
            cell_code           :       a string, the code for the LSTM cell (just for graph naming)
        Return:
            the output tensor and variable list of this block
        """
        fc1, w_fc1, b_fc1 = new_fc_layer(self.inside_LSTM_nn_placeholder, self.nn_inside_LSTM_inputfeature_num, layer_out_num1,\
                                             name='fc1_nn' + nn_code +"_" + cell_code, activation="RELU")
        drop1 = tf.nn.dropout(fc1, 0.8)
        fc2, w_fc2, b_fc2 = new_fc_layer(drop1, layer_out_num1, layer_out_num2,\
                                             name='fc2_nn' + nn_code +"_" + cell_code, activation="RELU")
        drop2 = tf.nn.dropout(fc2, 0.8)
        fc3, w_fc3, b_fc3 = new_fc_layer(drop2, layer_out_num2, layer_out_num3,\
                                             name='fc3_nn' + nn_code +"_" + cell_code, activation="RELU")
        drop3 = tf.nn.dropout(fc3, 0.8)
        fc4, w_fc4, b_fc4 = new_fc_layer(drop3, layer_out_num3, layer_out_num4,\
                                             name='fc4_nn' + nn_code +"_" + cell_code, activation="RELU")
        drop4 = tf.nn.dropout(fc4)
        fc5, w_fc5, b_fc5 = new_fc_layer(drop4, layer_out_num4, layer_out_num5,\
                                             name='fc5_nn' + nn_code +"_" + cell_code, activation="RELU")
        
        vars = [w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3, w_fc4, b_fc4, w_fc5, b_fc5]
        return fc5, vars


    def build_lstm_cell1(self, cell_code):
        """
        A function of LSTM cell
        Args:
            cell_code           :       a string, the code for the LSTM cell (just for graph naming)
        Return:
            the output tensor, feature memory and variable list of this LSTM cell
        """
        s1 = tf.concat([self.input_feature_placeholder, self.output_feature_placeholder], axis=1, name='concat_' + str(cell_code))
        s2, s2_vars = inside_LSTM_nn(10, 10, 10, 10, self.nn1_inside_LSTM_outputfeature_num, '1', cell_code)
        s3, s3_vars = inside_LSTM_nn(10, 10, 10, 10, self.nn2_inside_LSTM_outputfeature_num, '2', cell_code)
        s4, s4_vars = inside_LSTM_nn(10, 10, 10, 10, self.nn3_inside_LSTM_outputfeature_num, '3', cell_code)
        s4 = tf.tanh(s4, name='tanh_s4_' + cell_code)
        s5, s5_vars = inside_LSTM_nn(10, 10, 10, 10, self.nn4_inside_LSTM_outputfeature_num, '4', cell_code)
        s6 = tf.multiply(s3, s4, name = 'multiply_s3_s4_' + cell_code)
        s7 = tf.multiply(self.memory_feature_num, s2, name = 'multiply_s2_memory_' + cell_code)
        s8 = tf.add(s6, s7, name='add_s6_s7_' + cell_code)
        s9 = tf.tanh(s8, name='tanh_s8_' + cell_code)
        s10 = tf.multiply(s5, s9, name = 'multiply_s5_s10_' + cell_code)

        out, w_out, b_out = new_fc_layer(s10, self.memory_feature_num, self.output_feature_num,\
                                             name='out_nn_' + cell_code, activation="none")
        s10_vars = [w_out, b_out]
        LSTM_vars =  list(chain(*[s2_vars, s3_vars, s4_vars, s5_vars, s10_vars]))
        return out, s8, LSTM_vars

        
