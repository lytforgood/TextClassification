#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
1、自定义模型 Conv-BiGRU 卷积和循环并行
2、自定义模型 卷积和循环串行
"""
from keras.preprocessing import sequence
from keras.layers import Dense, Input, Flatten,Permute,Reshape
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.layers import merge
from keras.models import Model
from keras import backend as K
from keras.models import Sequential

import numpy as np
import pandas as pd
import cPickle as pickle
import numpy as np
import gensim

##数据获取
print('Loading data...')
path='./data/nlpmaildatasample2.pkl'
f2 = file(path, 'rb')
d = pickle.load(f2)
f2.close()
# path='./data/nlpmaildatasample2.csv'
# d = pd.read_csv(path,header=None)
# d.columns=['title','lable']

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
hidden_dim_1 = 200
hidden_dim_2 = 100
w2vpath="./data/w2c_model"

x_train, y_train, x_test, y_test = dx[0:len(dx)/5*3],dy[0:len(dx)/5*3],dx[len(dx)/5*3:len(dx)],dy[len(dx)/5*3:len(dx)]
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


print('Indexing word vectors.')
embeddings_index = {}
model = gensim.models.Word2Vec.load(w2vpath)
for word in words:
    embeddings_index[word]=model[word]
print('Found %s word vectors.' % len(embeddings_index))

print('Preparing embedding matrix.')
max_token = min(max_features, len(word_to_id))
embedding_matrix = np.zeros((max_token + 1, embedding_dims))
for word, i in word_to_id.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector # word_index to word_embedding_vector ,<20000(max_token)

embedding_layer = Embedding(max_token+1,embedding_dims,input_length=maxlen,weights=[embedding_matrix],trainable=False)

####并行
model_left = Sequential()
model_left.add(embedding_layer)
model_left.add(Bidirectional(GRU(128)))

model_right = Sequential()
model_right.add(embedding_layer)
model_right.add(Conv1D(128, 5, activation='relu')) #128卷积核的个数 5卷积核大小
model_right.add(MaxPooling1D())#5
model_right.add(Conv1D(128, 1, activation='relu'))
model_right.add(MaxPooling1D())#5
model_right.add(Flatten())

merged = Merge([model_left, model_right], mode='concat')
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

####串行
sentence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
conv_1=Conv1D(128, 3, activation='relu')(embedded_sequences)
maxpool_1=MaxPooling1D()(conv_1)
drop_1 = Dropout(0.2)(maxpool_1)
biGRU=Bidirectional(GRU(128))(drop_1)
drop_2 = Dropout(0.5)(biGRU)
dense_1 = Dense(1, activation='relu')(drop_2)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
