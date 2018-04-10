#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
未使用word2vector的双向lstm
t 时刻输出不仅取决于之前时刻的序列输入，还取决于将来时刻序列输入
embedding--->bi-directional lstm--->concat output--->average----->softmax
lstm中的Xt-1,Xt代表的是一个样本中的每一个词 所有循环只在一个样本中循环
TimeDistributed包装器=把一个层应用到输入的每一个时间步上-http://keras-cn.readthedocs.io/en/latest/layers/wrapper/
思考：
分类的时候不只使用最后一个隐藏元的输出，而是把所有隐藏元的输出做K-MaxPooling再分类
在双向GRU前添加单层卷积层提取一次ngram特征-C-GRU
"""
from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb

import pandas as pd
import cPickle as pickle
import numpy as np
import gensim

##数据获取
print('Loading data...')
path='./data/nlpmaildatasample2.csv'
d = pd.read_csv(path,header=None)
d.columns=['title','lable']

all_data=set()
for line in d["title"]:
   ws=line.split(" ")
   for w in ws:
     if w == ' ' or w == '' or w=="\t":
        continue
     all_data.add(w)
words=list(all_data)
word_to_id = dict(zip(words, range(len(words))))
dx=[]
for line in d["title"]:
    ws=line.split(" ")
    dx.append([word_to_id[w] for w in ws if w in word_to_id])
# dy=list(d['lable'])
dy=d['lable']

# set parameters:
maxlen=np.max(list(map(len, dx))) #maxlen = 400  最长文本词数
max_features = len(word_to_id)+1
batch_size = 32
embedding_dims=128

x_train, y_train, x_test, y_test = dx[0:len(dx)/5*3],dy[0:len(dx)/5*3],dx[len(dx)/5*3:len(dx)],dy[len(dx)/5*3:len(dx)]
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Bidirectional(LSTM(64))) ### 输出维度64 GRU
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))
# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
#lstm常选参数model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# a stateful LSTM model
#lahead: the input sequence length that the LSTM
# https://github.com/keras-team/keras/blob/master/examples/lstm_stateful.py
# model = Sequential()
# model.add(LSTM(20,input_shape=(lahead, 1),
#               batch_size=batch_size,
#               stateful=stateful))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')


print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[x_test, y_test])

# y_pred = model.predict_classes(x_test, verbose=0)
