# A Deep-Learning-Based Chinese Speech Recognition System


## 相关资料 

原理请查看本文：
* [ASRT：一个中文语音识别系统](https://blog.ailemon.me/2018/08/29/asrt-a-chinese-speech-recognition-system/)

训练和部署教程请看：
* [教你如何使用ASRT训练中文语音识别模型](<https://blog.ailemon.me/2020/08/20/teach-you-how-use-asrt-train-chinese-asr-model/>)
* [教你如何使用ASRT部署中文语音识别API服务器](<https://blog.ailemon.me/2020/08/27/teach-you-how-use-asrt-deploy-chinese-asr-api-server/>)

关于经常被问到的统计语言模型原理的问题，请看：

* [统计语言模型：从中文拼音到文本](https://blog.ailemon.me/2017/04/27/statistical-language-model-chinese-pinyin-to-words/)
* [统计N元语言模型生成算法：简单中文词频统计](https://blog.ailemon.me/2017/02/20/simple-words-frequency-statistic-without-segmentation-algorithm/)

关于CTC的问题请看：

* [[翻译]使用CTC进行序列建模](<https://blog.ailemon.me/2019/07/18/sequence-modeling-with-ctc/>)



## Introduction 简介


* **操作步骤**

首先通过Git将本项目克隆到您的计算机上，然后下载本项目训练所需要的数据集，下载链接详见[文档末尾部分](https://github.com/nl8590687/ASRT_SpeechRecognition#data-sets-%E6%95%B0%E6%8D%AE%E9%9B%86)。
```shell
$ git clone https://github.com/nl8590687/ASRT_SpeechRecognition.git
```

或者您也可以通过 "Fork" 按钮，将本项目Copy一份副本，然后通过您自己的SSH密钥克隆到本地。

通过git克隆仓库以后，进入项目根目录；并创建子目录 `dataset/` (可使用软链接代替)，然后将下载好的数据集直接解压进去

注意，当前版本中，Thchs30和ST-CMDS两个数据集都必须下载使用，缺一不可，并且使用其他数据集需要修改代码。

```shell
$ cd ASRT_SpeechRecognition

$ mkdir dataset

$ tar zxf <数据集压缩文件名> -C dataset/ 
```

然后需要将datalist目录下的文件全部拷贝到 `dataset/` 目录下，也就是将其跟数据集放在一起。
```shell
$ cp -rf datalist/* dataset/
```

目前可用的模型有24、25和251

运行本项目之前，请安装必要的[Python3版依赖库](https://github.com/nl8590687/ASRT_SpeechRecognition#python-import)

本项目开始训练请执行：
```shell
$ python3 train_mspeech.py
```
本项目开始测试请执行：
```shell
$ python3 test_mspeech.py
```
测试之前，请确保代码中填写的模型文件路径存在。

ASRT API服务器启动请执行：
```shell
$ python3 asrserver.py
```

如果要训练和使用非251版模型，请在代码中 `import SpeechModel` 的相应位置做修改。

## Model 模型

### Speech Model 语音模型

CNN + LSTM/GRU + CTC

其中，输入的音频的最大时间长度为16秒，输出为对应的汉语拼音序列

* 关于下载已经训练好的模型的问题

已经训练好的模型包含在发布版服务端程序压缩包里面，发布版成品服务端程序可以在此下载：[下载页面](https://asrt.ailemon.net/download)。

Github本仓库下[Releases](https://github.com/nl8590687/ASRT_SpeechRecognition/releases)页面里面还包括各个不同版本的介绍信息，每个版本下方的zip压缩包也是包含已经训练好的模型的发布版服务端程序压缩包。

### Language Model 语言模型

基于概率图的最大熵隐马尔可夫模型

输入为汉语拼音序列，输出为对应的汉字文本

## About Accuracy 关于准确率

当前，最好的模型在测试集上基本能达到80%的汉语拼音正确率

不过由于目前国际和国内的部分团队能做到98%，所以正确率仍有待于进一步提高

## Python Import
Python的依赖库

* python_speech_features
* TensorFlow (1.14 - 2.x)
* Numpy
* wave
* matplotlib
* math
* Scipy
* h5py
* http
* urllib
* requests

不会安装环境的同学请直接运行以下命令(前提是有GPU且已经安装好 CUDA 10.0 和 cudnn 7.4)：

```shell
$ pip install -r requirements.txt
```

[程序运行依赖环境详细说明](https://asrt.ailemon.net/docs/dependent-environment)

## Data Sets 数据集

[几个最新免费开源的中文语音数据集](https://blog.ailemon.me/2018/11/21/free-open-source-chinese-speech-datasets/)

* **清华大学THCHS30中文语音数据集**

  data_thchs30.tgz 
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/18/data_thchs30.tgz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/18/data_thchs30.tgz>)

  test-noise.tgz 
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/18/test-noise.tgz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/18/test-noise.tgz>)

  resource.tgz 
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/18/resource.tgz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/18/resource.tgz>)

* **Free ST Chinese Mandarin Corpus** 

  ST-CMDS-20170001_1-OS.tar.gz 
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/38/ST-CMDS-20170001_1-OS.tar.gz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz>)

* **AIShell-1 开源版数据集** 

  data_aishell.tgz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/33/data_aishell.tgz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/33/data_aishell.tgz>)

  注：数据集解压方法

  ```
  $ tar xzf data_aishell.tgz
  $ cd data_aishell/wav
  $ for tar in *.tar.gz;  do tar xvf $tar; done
  ```

* **Primewords Chinese Corpus Set 1** 

  primewords_md_2018_set1.tar.gz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/47/primewords_md_2018_set1.tar.gz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/47/primewords_md_2018_set1.tar.gz>)

* **aidatatang_200zh**

   aidatatang_200zh.tgz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/62/aidatatang_200zh.tgz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/62/aidatatang_200zh.tgz>)

* **MagicData**

  train_set.tar.gz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/68/train_set.tar.gz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/68/train_set.tar.gz>)

  dev_set.tar.gz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/68/dev_set.tar.gz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/68/dev_set.tar.gz>)

  test_set.tar.gz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/68/test_set.tar.gz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/68/test_set.tar.gz>)

  metadata.tar.gz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/68/metadata.tar.gz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/68/metadata.tar.gz>)

特别鸣谢！感谢前辈们的公开语音数据集

如果提供的数据集链接无法打开和下载，请点击该链接 [OpenSLR](http://www.openslr.org)

