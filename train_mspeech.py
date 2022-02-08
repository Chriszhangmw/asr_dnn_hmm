
import platform as plat
import os

import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session


from SpeechModel251 import ModelSpeech, ModelName
# tf.device('/gpu:3')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#进行配置，使用95%的GPU
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)

datapath = ''
modelpath = 'model_speech'


if(not os.path.exists(modelpath)): # 判断保存模型的目录是否存在
	os.makedirs(modelpath) # 如果不存在，就新建一个，避免之后保存模型的时候炸掉
	os.makedirs(modelpath + '/m' + ModelName)
datapath = 'dataset'
modelpath = modelpath + '/'

ms = ModelSpeech(datapath)

#ms.LoadModel(modelpath + 'speech_model251_e_0_step_327500.h5')
ms.TrainModel(datapath, epoch = 200, batch_size = 32)


