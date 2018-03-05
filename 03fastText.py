#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
文本分类之fastText
方法一：自己编写
方法二：Facebook开源工具https://github.com/facebookresearch/fastText#text-classification
paper:https://arxiv.org/pdf/1607.01759.pdf
fastText的核心思想就是：将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类
字符级n-gram特征的引入以及分层Softmax分类
参考：
http://blog.csdn.net/sinat_26917383/article/details/54850933
http://www.52nlp.cn/category/text-classification
"""
#方法二 fastText对词向量生成考虑到上下文  基于Hierarchical(分层) Softmax
# 输入格式  词(空格分开)_lable_标签  eg：英媒 称 威 __label__affairs
import pandas as pd
import re
import jieba
import cPickle as pickle
import numpy as np

##读取文件
path='./data/nlpmaildatasample2.csv'
d = pd.read_csv(path,header=None)
d.columns=['title','lable']

dtrain=d[0:d.shape[0]/5*3]
dtest=d[d.shape[0]/5*3:d.shape[0]]

#生成训练文件
def w2file(data,filename):
    f = open(filename,"w")
    for i in range(data.shape[0]):
        outline = d['title'][i] + "\t__label__" + str(d['lable'][i]) + "\n"
        f.write(outline)
    f.close()

w2file(dtrain,"./data/fasttext_train.txt")
w2file(dtest,"./data/fasttext_test.txt")

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fastText
#训练模型
classifier = fastText.FastText.train_supervised("./data/fasttext_train.txt",lr=0.1, dim=100,wordNgrams=1,label=u"__label__")
#参数
# train_supervised(input, lr=0.1, dim=100, ws=5, epoch=5, minCount=1, minCountLabel=0, minn=0, maxn=0, neg=5, wordNgrams=1, loss=u'softmax', bucket=2000000, thread=12, lrUpdateRate=100, t=0.0001, label=u'__label__', verbose=2, pretrainedVectors=u'')
# input_file     training file path (required)
# output         output file path (required)
# lr             learning rate [0.05]
# lr_update_rate change the rate of updates for the learning rate [100]
# dim            size of word vectors [100]
# ws             size of the context window [5]
# epoch          number of epochs [5]
# min_count      minimal number of word occurences [5]
# neg            number of negatives sampled [5]
# word_ngrams    max length of word ngram [1]
# loss           loss function {ns, hs, softmax} [ns]
# bucket         number of buckets [2000000]
# minn           min length of char ngram [3]
# maxn           max length of char ngram [6]
# thread         number of threads [12]
# t              sampling threshold [0.0001]
# silent         disable the log output from the C++ extension [1]
# encoding       specify input_file encoding [utf-8]
((u'__label__0',), array([ 0.77616984]))
#测试模型 help(classifier)
result = classifier.test("./data/fasttext_test.txt")
print result
texts=[str(t).decode("utf-8") for t in dtest["title"]] #预测与输入编码必须一致
##predict输出格式((u'__label__0',), array([ 0.77616984]))
y_pred = [int(e[0].replace("__label__","")) for e in classifier.predict(texts)[0]] #预测输出结果为元组
y_test=list(dtest["lable"])
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
print("Accuracy: %0.2f" % accuracy_score(y_test, y_pred))
print("F1: %0.2f" % f1_score(y_test, y_pred))
# Accuracy: 0.73
# F1: 0.65
