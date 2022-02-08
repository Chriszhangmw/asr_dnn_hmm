
import platform as plat
import Levenshtein as Lev
from SpeechModel251 import ModelSpeech
from LanguageModel2 import ModelLanguage
from tensorflow.keras import backend as K
import os
import time
import datetime

import timeout_decorator

def cal_cer(s1,s2):
	s1, s2, = s1.replace(" ", ""), s2.replace(" ", "")
	return Lev.distance(s1, s2)


@timeout_decorator.timeout(10)
def predict(model,ml,path):
	r = model.RecognizeSpeech_FromFile(path)
	# K.clear_session()
	r = ml.SpeechToText(r)
	return r

def cal_test(model,ml,testIndexPath):
	cer = 0
	with open(testIndexPath, 'r', encoding='utf-8') as f:
		idx = f.readlines()
		f.close()
	number = 0
	for line in idx:
		line = line.strip().split(',')
		path = line[0]
		label = line[1]
		try:
			r = predict(model,ml,path)
		except:
			continue
		print('*' * 20)
		print('源标签：', label)
		print('预测值：', r)
		v1 = float(cal_cer(r, label))
		print('CER 值： ', v1/float(len(label)))
		cer += v1 / float(len(label))
		number += 1
	cer = cer / float(number)
	print('total testing sample: ',number)
	print("corrected CER :", cer)

if __name__ == "__main__":
	datapath = ''
	modelpath = 'model_speech'
	os.environ["CUDA_VISIBLE_DEVICES"] = "3"
	datapath = 'dataset'
	modelpath = modelpath + '/'
	ms = ModelSpeech(datapath)
	ms.LoadModel(modelpath + '/m251/speech_model251best.h5')
	ml = ModelLanguage('./model_language')
	ml.LoadModel()
	cal_test(ms,ml,'./dataset/test/test.index')
	# r = ms.RecognizeSpeech_FromFile('./dataset/data/D8_982.wav')



	# print('*[提示] 语音识别结果：\n', r)



	# str_pinyin = r
	# # str_pinyin =  ['su1', 'bei3', 'jun1', 'de5', 'yi4','xie1', 'ai4', 'guo2', 'jiang4', 'shi4', 'ma3', 'zhan4', 'shan1', 'ming2', 'yi1', 'dong4', 'ta1', 'ju4', 'su1', 'bi3', 'ai4', 'dan4', 'tian2','mei2', 'bai3', 'ye3', 'fei1', 'qi3', 'kan4', 'zhan4']
	# r = ml.SpeechToText(str_pinyin)
	# print('语音转文字结果：\n', r)












