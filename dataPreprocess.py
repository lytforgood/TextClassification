# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gensim
from keras.preprocessing.sequence import pad_sequences

"""
dataPreprocess
set
    path='./data/nlpmail_re3.txt'
    batch_size = 32
    embedding_dims = 128  #词向量长度
    epochs = 100
    w2vpath="./data/w2c_model"
    hidden_dim_1 = 200
    hidden_dim_2 = 100
return x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,
"""
def getdata(path,embedding_dims,w2vpath):
    print('Loading data...')
    d = pd.read_csv(path,header=None)
    d.columns=['title','lable']

    #drop=True 不生成index列
    d=d[-pd.isnull(d["title"])].reset_index(drop=True)

    all_data=set()
    for line in d["title"]:
       ws=str(line).split(" ")
       for w in ws:
         if w == ' ' or w == '' or w=="\t" or w=="??":
            continue
         all_data.add(w)
    words=list(all_data)

    word_to_id = dict(zip(words, range(len(words))))
    dx=[]
    for line in d["title"]:
        ws=str(line).split(" ")
        dx.append([word_to_id[w] for w in ws if w in word_to_id])
    # dy=list(d['lable'])
    dy=d['lable']

    print('Average  sequence length: {}'.format(np.mean(list(map(len, dx)), dtype=int)))
    # set parameters:
    maxlen=np.max(list(map(len, dx))) #maxlen = 29  最长文本词数

    inx=int(len(dx)/5*3)
    x_train, y_train, x_test, y_test = dx[0:inx],dy[0:inx],dx[inx:len(dx)],dy[inx:len(dx)]

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

    #初始化一个0向量 统计未出现词个数
    null_word=np.zeros(embedding_dims)
    null_word_count=0

    for word in words:
        try:
            embeddings_index[word]=model[word]
        except:
            embeddings_index[word]=null_word
            null_word_count+=1
    print('Found %s word vectors.' % len(embeddings_index))
    print('Found %s null word.' % null_word_count)

    print('Preparing embedding matrix.')
    max_token = len(word_to_id)
    embedding_matrix = np.zeros((max_token + 1, embedding_dims))
    for word, i in word_to_id.items():
        if i > max_token:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix