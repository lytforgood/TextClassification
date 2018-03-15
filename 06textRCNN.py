#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
使用Word2vec定义词向量矩阵
recurrent structure (convolutional layer)：
词向量矩阵
left(无意义补0+去最后一个词)  max_token对应词向量为0向量
right(去第一个词+无意义补0)
lstm(left)+词向量矩阵+lstm(right)===上一个词+当前词+下一个词
structure:1)recurrent structure (convolutional layer) 2)max pooling 3) fully connected layer+softmax
Recurrent convolutional neural networks for text classification
http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745
tensoflow版https://github.com/brightmart/text_classification/blob/master/a04_TextRCNN/p71_TextRCNN_model.py
"""
import pandas as pd
import cPickle as pickle
import numpy as np
import gensim
from keras.preprocessing import sequence
from keras import backend
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model

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

print('Build model...')
document = Input(shape = (None, ), dtype = "int32")
left_context = Input(shape = (None, ), dtype = "int32")
right_context = Input(shape = (None, ), dtype = "int32")

embedder = Embedding(max_token + 1, embedding_dims, weights = [embedding_matrix], trainable = False)
doc_embedding = embedder(document)
l_embedding = embedder(left_context)
r_embedding = embedder(right_context)

# I use LSTM RNNs instead of vanilla RNNs as described in the paper.
forward = LSTM(hidden_dim_1, return_sequences = True)(l_embedding) # See equation (1).
backward = LSTM(hidden_dim_1, return_sequences = True, go_backwards = True)(r_embedding) # See equation (2).
together = concatenate([forward, doc_embedding, backward], axis = 2) # See equation (3).

semantic = TimeDistributed(Dense(hidden_dim_2, activation = "tanh"))(together) # See equation (4).

# Keras provides its own max-pooling layers, but they cannot handle variable length input
# (as far as I can tell). As a result, I define my own max-pooling layer here.
pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (hidden_dim_2, ))(semantic) # See equation (5).

output = Dense(1, input_dim = hidden_dim_2, activation = "softmax")(pool_rnn) # See equations (6) and (7).NUM_CLASSES=1

model = Model(inputs = [document, left_context, right_context], outputs = output)
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

##生成左右上下文
print('Build left and right data')
doc_x_train = np.array(x_train)
# We shift the document to the right to obtain the left-side contexts.
left_x_train = np.array([[max_token]+t_one[:-1].tolist() for t_one in x_train])
# We shift the document to the left to obtain the right-side contexts.
right_x_train = np.array([t_one[1:].tolist()+[max_token] for t_one in x_train])

doc_x_test = np.array(x_test)
# We shift the document to the right to obtain the left-side contexts.
left_x_test = np.array([[max_token]+t_one[:-1].tolist() for t_one in x_test])
# We shift the document to the left to obtain the right-side contexts.
right_x_test = np.array([t_one[1:].tolist()+[max_token] for t_one in x_test])


# history = model.fit([doc_x_train, left_x_train, right_x_train], y_train, epochs = 1)
# loss = history.history["loss"][0]
model.fit([doc_x_train, left_x_train, right_x_train], y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[[doc_x_test, left_x_test, right_x_test], y_test])

