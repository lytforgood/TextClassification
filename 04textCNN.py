#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
文本分类之textCNN  疑问MaxPooling1D？使用计算？区别和GlobalMaxPooling1D
论文：Convolutional Neural Networks for Sentence Classification
论文解读：http://www.jeyzhang.com/cnn-apply-on-modelling-sentence.html
输入层：词个数x词向量维数---矩阵的类型可以是静态的(static)word vector是固定不变，动态的(non static)word vector也当做是可优化的参数这一过程称为 Fine tune
卷积层：若干个Feature Map--不同大小滤波器 卷积核大小为nxk k是词向量维度 1D默认宽度为词向量维度
池化层：Max-over-time Pooling--输出为各个Feature Map的最大值们，即一个一维的向量
全连接 + Softmax层：池化层的一维向量的输出通过全连接的方式，连接一个Softmax层
Dropout：倒数第二层的全连接部分，L2正则化，减轻过拟合
词向量变种：
CNN-rand：对不同单词的向量作随机初始化，BP的时候作调整  Embedding层选择随机初始化方法
static：拿word2vec, FastText or GloVe训练好的词向量
non-static：拿word2vec, FastText or GloVe训练好的词向量，训练过程中再对它们微调Fine tuned(自己理解：先用其他大文本语料训练w2v再用本文本训练w2v)
multiple channel ：类比于图像中的RGB通道, 这里也可以用 static 与 non-static 搭两个通道来搞
结论：
CNN-static较与CNN-rand好，说明pre-training的word vector确实有较大的提升作用（这也难怪，因为pre-training的word vector显然利用了更大规模的文本数据信息）；
CNN-non-static较于CNN-static大部分要好，说明适当的Fine tune也是有利的，是因为使得vectors更加贴近于具体的任务；
CNN-multichannel较于CNN-single在小规模的数据集上有更好的表现，实际上CNN-multichannel体现了一种折中思想，即既不希望Fine tuned的vector距离原始值太远，但同时保留其一定的变化空间
github：https://github.com/yoonkim/CNN_sentence
code参考
http://blog.csdn.net/diye2008/article/details/53105652?locationNum=11&fps=1
glove embedding参考http://blog.csdn.net/sscssz/article/details/53333225
"""
from __future__ import print_function

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten,GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding,Dropout
from keras.models import Model
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import merge
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


print('Average  sequence length: {}'.format(np.mean(list(map(len, dx)), dtype=int)))

# set parameters:
maxlen=np.max(list(map(len, dx))) #maxlen = 400  最长文本词数
max_features = 20000  #字典允许最大大小
batch_size = 32
embedding_dims = 64  #词向量长度
epochs = 2
w2vpath="./data/w2c_model"

x_train, y_train, x_test, y_test = dx[0:len(dx)/5*3],dy[0:len(dx)/5*3],dx[len(dx)/5*3:len(dx)],dy[len(dx)/5*3:len(dx)]
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


print('Indexing word vectors.')
embeddings_index = {}
model = gensim.models.Word2Vec.load(w2vpath)
for word in words:
    embeddings_index[word]=model[word]
print('Found %s word vectors.' % len(embeddings_index))

print('Preparing embedding matrix.')
nb_words = min(max_features, len(word_to_id))
embedding_matrix = np.zeros((nb_words + 1, embedding_dims))
for word, i in word_to_id.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector # word_index to word_embedding_vector ,<20000(nb_words)


# 神经网路的第一层，词向量层，本文使用了预训练word2vec词向量，可以把trainable那里设为False
embedding_layer = Embedding(nb_words+1,
                            embedding_dims,
                            input_length=maxlen,
                            weights=[embedding_matrix],
                            trainable=False)
print('Build model...')
##最简单cnn
# model = Sequential()
# model.add(Embedding(nb_words + 1,
#                     embedding_dims,
#                     input_length=maxlen))
# model.add(Dropout(0.2))
# model.add(Conv1D(250,#filters
#                  3,#kernel_size
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(250))#hidden layer:
# model.add(Dropout(0.2))
# model.add(Activation('relu'))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_data=(x_test, y_test))

###3层合并model 经过词向量表达的文本为一维数据，因此在TextCNN卷积用的是一维卷积
#left model
model_left = Sequential()
#https://keras.io/layers/embeddings/
# model.add(Embedding(max_features,embedding_dims,input_length=maxlen))
model_left.add(embedding_layer)
model_left.add(Conv1D(128, 5, activation='relu')) #128输出的维度 5卷积核大小
model_left.add(MaxPooling1D())#5
model_left.add(Conv1D(128, 5, activation='relu'))
model_left.add(MaxPooling1D())#5
model_left.add(Conv1D(128, 5, activation='relu'))
model_left.add(MaxPooling1D()) #35 #model_left.add(GlobalMaxPooling1D())
model_left.add(Flatten())

model_right = Sequential()
model_right.add(embedding_layer)
model_right.add(Conv1D(128, 4, activation='relu'))
model_right.add(MaxPooling1D())#4
model_right.add(Conv1D(128, 4, activation='relu'))
model_right.add(MaxPooling1D())#4
model_right.add(Conv1D(128, 4, activation='relu'))
model_right.add(MaxPooling1D())#28
model_right.add(Flatten())

model_3 = Sequential()
model_3.add(embedding_layer)
model_3.add(Conv1D(128, 6, activation='relu'))
model_3.add(MaxPooling1D())#3
model_3.add(Conv1D(128, 6, activation='relu'))
model_3.add(MaxPooling1D())#3
model_3.add(Conv1D(128, 6, activation='relu'))
model_3.add(MaxPooling1D())#30
model_3.add(Flatten())

merged = Merge([model_left, model_right,model_3], mode='concat') # 将三种不同卷积窗口的卷积层组合 连接在一起，当然也可以只是用三个model中的一个，一样可以得到不错的效果，只是本文采用论文中的结构设计
model = Sequential()
model.add(merged) # add merge
model.add(Dense(128, activation='relu')) # 全连接层
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax')) # softmax，输出文本属于类别中每个类别的概率

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

score = model.evaluate(x_train, y_train, verbose=0) # 评估模型在训练集中的效果，准确率约99%
print('train score:', score[0])
print('train accuracy:', score[1])
score = model.evaluate(x_test, y_test, verbose=0)  # 评估模型在测试集中的效果，准确率约为97%，迭代次数多了，会进一步提升
print('Test score:', score[0])
print('Test accuracy:', score[1])
