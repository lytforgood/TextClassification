# -*- coding: utf-8 -*-
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding,Dropout
from keras.models import Model
from keras.optimizers import *
from keras.layers import concatenate
from keras.callbacks import EarlyStopping
import logging

def train(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,modelname):
    print(modelname + 'Build model...')
    sentence = Input(shape = (None, ), dtype = "int32")
    embedding_layer = Embedding(max_token+1,
                                embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=False)
    sentence_embedding = embedding_layer(sentence)
    c2 = Conv1D(2, 2, activation='sigmoid')(sentence_embedding)
    p2 = MaxPooling1D(27)(c2)
    p2 = Flatten()(p2)

    c3 = Conv1D(2, 3, activation='sigmoid')(sentence_embedding)
    p3 = MaxPooling1D(26)(c3)
    p3 = Flatten()(p3)

    c4 = Conv1D(2, 4, activation='sigmoid')(sentence_embedding)
    p4 = MaxPooling1D(25)(c4)
    p4 = Flatten()(p4)

    x = concatenate([p2, p3, p4])
    output = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = sentence, outputs = output)
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

    # print(model.summary())
    #patience经过几个epoch后loss不在变化停止训练
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
    print('Train...')
    hist = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),callbacks=[early_stopping])

    # print(hist.history)
    ##输出loss与acc到日志文件
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=logpath, level=logging.DEBUG, format=log_format)
    logging.warning(modelname)
    for i in range(len(hist.history["acc"])):
        strlog=str(i+1)+" Epoch "+"-loss: "+str(hist.history["loss"][i])+" -acc: "+str(hist.history["acc"][i])+" -val_loss: "+str(hist.history["val_loss"][i])+" -val_acc: "+str(hist.history["val_acc"][i])
        logging.warning(strlog)

    model.save(modelpath + modelname + '.h5')