# -*- coding: utf-8 -*-
from keras import backend
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model

from keras.callbacks import EarlyStopping
import logging
import numpy as np


def train(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,modelname,hidden_dim_1,hidden_dim_2):
    print(modelname + 'Build model...')
    document = Input(shape=(None,), dtype="int32")
    left_context = Input(shape=(None,), dtype="int32")
    right_context = Input(shape=(None,), dtype="int32")

    embedder = Embedding(max_token + 1, embedding_dims, weights=[embedding_matrix], trainable=False) #input_length=maxlen
    doc_embedding = embedder(document)
    l_embedding = embedder(left_context)
    r_embedding = embedder(right_context)

    # I use LSTM RNNs instead of vanilla RNNs as described in the paper.
    forward = LSTM(hidden_dim_1, return_sequences=True)(l_embedding)  # See equation (1).
    backward = LSTM(hidden_dim_1, return_sequences=True, go_backwards=True)(r_embedding)  # See equation (2).
    together = concatenate([forward, doc_embedding, backward], axis=2)  # See equation (3).

    semantic = TimeDistributed(Dense(hidden_dim_2, activation="tanh"))(together)  # See equation (4).

    # Keras provides its own max-pooling layers, but they cannot handle variable length input
    # (as far as I can tell). As a result, I define my own max-pooling layer here.
    pool_rnn = Lambda(lambda x: backend.max(x, axis=1), output_shape=(hidden_dim_2,))(semantic)  # See equation (5).

    output = Dense(1, input_dim=hidden_dim_2, activation="sigmoid")(pool_rnn)  # See equations (6) and (7).NUM_CLASSES=1

    model = Model(inputs=[document, left_context, right_context], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    ##生成左右上下文
    print('Build left and right data')
    doc_x_train = np.array(x_train)
    # We shift the document to the right to obtain the left-side contexts.
    left_x_train = np.array([[max_token] + t_one[:-1].tolist() for t_one in x_train])
    # We shift the document to the left to obtain the right-side contexts.
    right_x_train = np.array([t_one[1:].tolist() + [max_token] for t_one in x_train])

    doc_x_test = np.array(x_test)
    # We shift the document to the right to obtain the left-side contexts.
    left_x_test = np.array([[max_token] + t_one[:-1].tolist() for t_one in x_test])
    # We shift the document to the left to obtain the right-side contexts.
    right_x_test = np.array([t_one[1:].tolist() + [max_token] for t_one in x_test])

    # patience经过几个epoch后loss不在变化停止训练
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
    print('Train...')
    # history = model.fit([doc_x_train, left_x_train, right_x_train], y_train, epochs = 1)
    # loss = history.history["loss"][0]
    hist = model.fit([doc_x_train, left_x_train, right_x_train], y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=[[doc_x_test, left_x_test, right_x_test], y_test], callbacks=[early_stopping])

    # print(hist.history)
    ##输出loss与acc到日志文件
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=logpath, level=logging.DEBUG, format=log_format)
    logging.warning(modelname)
    for i in range(len(hist.history["acc"])):
        strlog=str(i+1)+" Epoch "+"-loss: "+str(hist.history["loss"][i])+" -acc: "+str(hist.history["acc"][i])+" -val_loss: "+str(hist.history["val_loss"][i])+" -val_acc: "+str(hist.history["val_acc"][i])
        logging.warning(strlog)

    model.save(modelpath + modelname + '.h5')