
import platform as plat
import os
import time

from general_function.file_wav import *
from general_function.file_dict import *
from general_function.gen_func import *

# LSTM_CNN
import tensorflow as tf
import tensorflow.keras as kr
import numpy as np
import random

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, BatchNormalization # , Flatten
from tensorflow.keras.layers import Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D #, Merge
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adadelta, Adam

from readdata24 import DataSpeech

abspath = ''
ModelName='251'
#NUM_GPU = 2
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
class ModelSpeech(): # 语音模型类
	def __init__(self, datapath):
		'''
		初始化
		默认输出的拼音的表示大小是1428，即1427个拼音+1个空白块
		'''
		MS_OUTPUT_SIZE = 1428
		self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE # 神经网络最终输出的每一个字符向量维度的大小
		#self.BATCH_SIZE = BATCH_SIZE # 一次训练的batch
		self.label_max_string_length = 64
		self.AUDIO_LENGTH = 1600
		self.AUDIO_FEATURE_LENGTH = 200
		self._model, self.base_model = self.CreateModel()
		self.datapath = datapath
		self.slash='/' # 正斜杠
		if(self.slash != self.datapath[-1]): # 在目录路径末尾增加斜杠
			self.datapath = self.datapath + self.slash
	
		
	def CreateModel(self):
		'''
		定义CNN/LSTM/CTC模型，使用函数式模型
		输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
		隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2
		隐藏层：全连接层
		输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
		CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出
		
		'''
		
		input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))
		
		layer_h1 = Conv2D(32, (3,3), use_bias=False, activation='relu', padding='same', kernel_initializer='he_normal')(input_data) # 卷积层
		layer_h1 = Dropout(0.05)(layer_h1)
		layer_h2 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1) # 卷积层
		layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2) # 池化层
		#layer_h3 = Dropout(0.2)(layer_h2) # 随机中断部分神经网络连接，防止过拟合
		layer_h3 = Dropout(0.05)(layer_h3)
		layer_h4 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3) # 卷积层
		layer_h4 = Dropout(0.1)(layer_h4)
		layer_h5 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4) # 卷积层
		layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5) # 池化层
		
		layer_h6 = Dropout(0.1)(layer_h6)
		layer_h7 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h6) # 卷积层
		layer_h7 = Dropout(0.15)(layer_h7)
		layer_h8 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h7) # 卷积层
		layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8) # 池化层
		
		layer_h9 = Dropout(0.15)(layer_h9)
		layer_h10 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h9) # 卷积层
		layer_h10 = Dropout(0.2)(layer_h10)
		layer_h11 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h10) # 卷积层
		layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11) # 池化层
		
		layer_h12 = Dropout(0.2)(layer_h12)
		layer_h13 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h12) # 卷积层
		layer_h13 = Dropout(0.2)(layer_h13)
		layer_h14 = Conv2D(128, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h13) # 卷积层
		layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14) # 池化层
		
		layer_h16 = Reshape((200, 3200))(layer_h15) #Reshape层
		layer_h16 = Dropout(0.3)(layer_h16)
		layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16) # 全连接层
		layer_h17 = Dropout(0.3)(layer_h17)
		layer_h18 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_h17) # 全连接层
		
		y_pred = Activation('softmax', name='Activation0')(layer_h18)
		model_base  = Model(inputs = input_data, outputs = y_pred)
		
		labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')
		input_length = Input(name='input_length', shape=[1], dtype='int64')
		label_length = Input(name='label_length', shape=[1], dtype='int64')
		# Keras doesn't currently support loss funcs with extra parameters
		# so CTC loss is implemented in a lambda layer
		
		#layer_out = Lambda(ctc_lambda_func,output_shape=(self.MS_OUTPUT_SIZE, ), name='ctc')([y_pred, labels, input_length, label_length])#(layer_h6) # CTC
		loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
		model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
		
		model.summary()
		
		# clipnorm seems to speeds up convergence
		#sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
		#opt = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)
		opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
		#model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
		model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = opt)

		print('[*Info] Create Model Successful, Compiles Model Successful. ')
		return model, model_base
		
	def ctc_lambda_func(self, args):
		y_pred, labels, input_length, label_length = args
		y_pred = y_pred[:, :, :]
		return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
	
	def TrainModel(self, datapath, epoch = 2, save_step = 500, batch_size = 32,
				   filename = abspath + 'model_speech/m' + ModelName + '/speech_model'+ModelName):
		data=DataSpeech(datapath, 'train')
		best_cer = 100000.0
		
		yielddatas = data.data_genetator(batch_size, self.AUDIO_LENGTH)
		
		for epoch in range(epoch): # 迭代轮数
			print('[running] train epoch %d .' % epoch)
			n_step = 0 # 迭代数据数
			try:
				self._model.fit_generator(yielddatas, save_step)
				n_step += 1
			except StopIteration:
				print('[error] generator error. please check data format.')
				break

			cer = self.TestModel(self.datapath, str_dataset='dev', data_count = 1000)
			print('evaluating model , CER is :',cer,' at epoch: ',epoch)
			if cer < best_cer:
				self.SaveModel(comment='best')
				best_cer = cer
				
	def LoadModel(self,filename = abspath + 'model_speech/m'+ModelName+'/speech_model'+ModelName+'.model'):
		'''
		加载模型参数
		'''
		self._model.load_weights(filename)
		#self.base_model.load_weights(filename + '.base')

	def SaveModel(self,filename = abspath + 'model_speech/m'+ModelName+'/speech_model'+ModelName,comment=''):
		'''
		保存模型参数
		'''
		self._model.save_weights(filename + comment + '.model')
		self.base_model.save_weights(filename + comment + '.model.base')
		# 需要安装 hdf5 模块
		self._model.save(filename + comment + '.h5')
		self.base_model.save(filename + comment + '.base.h5')
		f = open('step'+ModelName+'.txt','w')
		f.write(filename+comment)
		f.close()

	def TestModel(self, datapath='', str_dataset='dev', data_count = 1000,
				  out_report = False, show_ratio = True, io_step_print = 10, io_step_file = 10):
		data=DataSpeech(self.datapath, str_dataset)
		#data.LoadDataList(str_dataset) 
		num_data = data.GetDataNum() # 获取数据的数量
		if(data_count <= 0 or data_count > num_data): # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
			data_count = num_data
		try:
			ran_num = random.randint(0,num_data - 1) # 获取一个随机数
			
			words_num = 0
			word_error_num = 0

			for i in range(data_count):
				data_input, data_labels = data.GetData((ran_num + i) % num_data)  # 从随机数开始连续向后取一定数量数据
				# 数据格式出错处理 开始
				# 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
				num_bias = 0
				while(data_input.shape[0] > self.AUDIO_LENGTH):
					print('*[Error]','wave data lenghth of num',(ran_num + i) % num_data, 'is too long.','\n A Exception raise when test Speech Model.')
					num_bias += 1
					data_input, data_labels = data.GetData((ran_num + i + num_bias) % num_data)  # 从随机数开始连续向后取一定数量数据
				# 数据格式出错处理 结束
				pre = self.Predict(data_input, data_input.shape[0] // 8)
				words_n = data_labels.shape[0] # 获取每个句子的字数
				words_num += words_n # 把句子的总字数加上
				print('*'*20)
				print('原始标签： ',data_labels)
				print('预测标签： ', pre)
				edit_distance = GetEditDistance(data_labels, pre) # 获取编辑距离
				if(edit_distance <= words_n): # 当编辑距离小于等于句子字数时
					word_error_num += edit_distance # 使用编辑距离作为错误字数
				else: # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
					word_error_num += words_n # 就直接加句子本来的总字数就好了
				
				# if((i % io_step_print == 0 or i == data_count - 1) and show_ratio == True):
				# 	#print('测试进度：',i,'/',data_count)
				# 	print('Test Count: ',i,'/',data_count)

			print('*[Test Result] Speech Recognition ' + str_dataset + ' set word error ratio: ', word_error_num / words_num * 100, '%')
			return  float(word_error_num / words_num)
			
		except StopIteration:
			print('[Error] Model Test Error. please check data format.')
			return 10000
	
	def Predict(self, data_input, input_len):
		batch_size = 1 
		in_len = np.zeros((batch_size),dtype = np.int32)
		in_len[0] = input_len
		x_in = np.zeros((batch_size, 1600, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
		for i in range(batch_size):
			x_in[i,0:len(data_input)] = data_input
		base_pred = self.base_model.predict(x = x_in)
		base_pred =base_pred[:, :, :]
		r = K.ctc_decode(base_pred, in_len, greedy = True, beam_width=100, top_paths=1)
		if(tf.__version__[0:2] == '1.'):
			r1 = r[0][0].eval(session=tf.compat.v1.Session())
		else:
			r1 = r[0][0].numpy()
		return r1[0]
	
	def RecognizeSpeech(self, wavsignal, fs):
		'''
		最终做语音识别用的函数，识别一个wav序列的语音
		'''
		data_input = GetFrequencyFeature3(wavsignal, fs)
		input_length = len(data_input)
		input_length = input_length // 8
		data_input = np.array(data_input, dtype = np.float)
		data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
		r1 = self.Predict(data_input, input_length)
		list_symbol_dic = GetSymbolList(self.datapath) # 获取拼音列表

		r_str=[]
		for i in r1:
			r_str.append(list_symbol_dic[i])
		
		return r_str
		pass
		
	def RecognizeSpeech_FromFile(self, filename):
		'''
		最终做语音识别用的函数，识别指定文件名的语音
		'''
		
		wavsignal,fs = read_wav_data(filename)
		
		r = self.RecognizeSpeech(wavsignal, fs)
		
		return r
		

		
	
		
	@property
	def model(self):
		'''
		返回keras model
		'''
		return self._model


if(__name__=='__main__'):
	
	#import tensorflow as tf
	#from keras.backend.tensorflow_backend import set_session
	#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	#进行配置，使用95%的GPU
	#config = tf.ConfigProto()
	#config.gpu_options.per_process_gpu_memory_fraction = 0.95
	#config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
	#set_session(tf.Session(config=config))


	datapath =  abspath + ''
	modelpath =  abspath + 'model_speech'


	if(not os.path.exists(modelpath)): # 判断保存模型的目录是否存在
		os.makedirs(modelpath) # 如果不存在，就新建一个，避免之后保存模型的时候炸掉
	datapath =  abspath + 'dataset'
	modelpath = modelpath + '/'
	ms = ModelSpeech(datapath)


	#ms.LoadModel(modelpath + 'm251/speech_model251_e_0_step_100000.h5')
	ms.TrainModel(datapath, epoch = 50, batch_size = 16, save_step = 500)

