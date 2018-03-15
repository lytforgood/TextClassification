#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
单双向lstm 之后加 + Attention   HAN模型
paper:Hierarchical Attention Networks for Document Classification
加入Attention之后最大的好处自然是能够直观的解释各个句子和词对分类类别的重要性
Structure:
1.embedding
2.Word Encoder: 词级双向GRU，以获得丰富的词汇表征
3.Word Attention:词级注意在句子中获取重要信息
4.Sentence Encoder: 句子级双向GRU，以获得丰富的句子表征
5.Sentence Attetion: 句级注意以获得句子中的重点句子
6.FC+Softmax
# HierarchicalAttention: 1.Word Encoder. 2.Word Attention. 3.Sentence Encoder 4.Sentence Attention 5.linear classifier. 2017-06-13
Attention层是一个MLP+softmax机制
code参考：https://github.com/richliao/textClassifier
https://github.com/philipperemy/keras-attention-mechanism
https://github.com/codekansas/keras-language-modeling/blob/master/keras_models.py
https://github.com/codekansas/keras-language-modeling
https://github.com/EdGENetworks/attention-networks-for-classification
https://github.com/brightmart/text_classification/tree/master/a05_HierarchicalAttentionNetwork
原理解说：https://www.zhihu.com/question/68482809/answer/268320399
"""
from keras.preprocessing import sequence
from keras.layers import Dense, Input, Flatten,Permute,Reshape
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.layers import merge
from keras.models import Model
from keras import backend as K

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


##句子最多几句
max_sents=1

embedding_layer = Embedding(max_token + 1,
                            embedding_dims,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=True)
#LSTM步长
TIME_STEPS=maxlen
SINGLE_ATTENTION_VECTOR = False
##不带别名的自编写Attention
# def attention_3d_block(inputs):
#     # inputs.shape = (batch_size, time_steps, input_dim)
#     input_dim = int(inputs.shape[2])
#     a = Permute((2, 1))(inputs)
#     a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
#     a = Dense(TIME_STEPS, activation='softmax')(a)
#     if SINGLE_ATTENTION_VECTOR:
#         a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
#         a = RepeatVector(input_dim)(a)
#     a_probs = Permute((2, 1), name='attention_vec')(a)
#     output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
#     return output_attention_mul
##使用多次attention需要新命名
def attention_3d_block2(inputs,new_layer_name):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name=new_layer_name+'_'+'dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name=new_layer_name+'_''attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name=new_layer_name+'_''attention_mul', mode='mul')
    return output_attention_mul

#单向LSTM之后加入Attention
# sentence_input = Input(shape=(maxlen,), dtype='int32')
# embedded_sequences = embedding_layer(sentence_input)
# lstm_out = LSTM(100, return_sequences=True)(embedded_sequences)
# attention_mul = attention_3d_block(lstm_out)
# attention_mul = Flatten()(attention_mul)
# output = Dense(1, activation='sigmoid')(attention_mul)
# model = Model(sentence_input, output)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, validation_data=(x_test, y_test),
#           nb_epoch=epochs, batch_size=batch_size)

#双向LSTM词encoder  输入是 词标签数组
sentence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
forward_rnn = LSTM(100, return_sequences=True)
backward_rnn = LSTM(100, return_sequences=True, go_backwards=True)
lstm_out_f_rnn = forward_rnn(embedded_sequences)
attention_f_mul = attention_3d_block2(lstm_out_f_rnn,"forward")
lstm_out_b_rnn = backward_rnn(embedded_sequences)
attention_b_mul = attention_3d_block2(lstm_out_b_rnn,"backward")
attention_mul=merge([attention_f_mul, attention_b_mul], mode='concat', concat_axis=-1)
attention_mul = Flatten()(attention_mul)
output = Dense(1, activation='sigmoid')(attention_mul)
model = Model(sentence_input, output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          nb_epoch=epochs, batch_size=batch_size)

####先词Attention再句Attention Hierarchical Attention Networks for Document Classification
#词encoder  输入是 词标签数组 未完待续
#句encoder  输入是 句子个数x词标签数组

