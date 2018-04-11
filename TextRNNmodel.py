# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,GRU,SimpleRNN
import logging
from keras.callbacks import EarlyStopping

def train(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,modelname):
    embedding_layer = Embedding(max_token + 1,
                                embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=False)
    print(modelname + 'Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(SimpleRNN(128, activation="relu"))
    # model.add(LSTM(128))
    # model.add(Bidirectional(LSTM(200))) ### 输出维度64 GRU
    # model.add(Bidirectional(GRU(64)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    # lstm常选参数model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # a stateful LSTM model
    # lahead: the input sequence length that the LSTM
    # https://github.com/keras-team/keras/blob/master/examples/lstm_stateful.py
    # model = Sequential()
    # model.add(LSTM(20,input_shape=(lahead, 1),
    #               batch_size=batch_size,
    #               stateful=stateful))
    # model.add(Dense(1))
    # model.compile(loss='mse', optimizer='adam')

    # patience经过几个epoch后loss不在变化停止训练
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
    print('Train...')
    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(x_test, y_test), callbacks=[early_stopping])
    # print(hist.history)
    ##输出loss与acc到日志文件
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=logpath, level=logging.DEBUG, format=log_format)
    logging.warning(modelname)
    for i in range(len(hist.history["acc"])):
        strlog=str(i+1)+" Epoch "+"-loss: "+str(hist.history["loss"][i])+" -acc: "+str(hist.history["acc"][i])+" -val_loss: "+str(hist.history["val_loss"][i])+" -val_acc: "+str(hist.history["val_acc"][i])
        logging.warning(strlog)

    model.save(modelpath + modelname + '.h5')

def train2(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,modelname):
    embedding_layer = Embedding(max_token + 1,
                                embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=False)
    print(modelname + 'Build model...')
    model = Sequential()
    model.add(embedding_layer)
    # model.add(SimpleRNN(128, activation="relu"))
    # model.add(LSTM(128))
    model.add(Bidirectional(LSTM(200))) ### 输出维度64 GRU
    # model.add(Bidirectional(GRU(64)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    # lstm常选参数model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # a stateful LSTM model
    # lahead: the input sequence length that the LSTM
    # https://github.com/keras-team/keras/blob/master/examples/lstm_stateful.py
    # model = Sequential()
    # model.add(LSTM(20,input_shape=(lahead, 1),
    #               batch_size=batch_size,
    #               stateful=stateful))
    # model.add(Dense(1))
    # model.compile(loss='mse', optimizer='adam')

    # patience经过几个epoch后loss不在变化停止训练
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
    print('Train...')
    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(x_test, y_test), callbacks=[early_stopping])
    # print(hist.history)
    ##输出loss与acc到日志文件
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=logpath, level=logging.DEBUG, format=log_format)
    logging.warning(modelname)
    for i in range(len(hist.history["acc"])):
        strlog=str(i+1)+" Epoch "+"-loss: "+str(hist.history["loss"][i])+" -acc: "+str(hist.history["acc"][i])+" -val_loss: "+str(hist.history["val_loss"][i])+" -val_acc: "+str(hist.history["val_acc"][i])
        logging.warning(strlog)

    model.save(modelpath + modelname + '.h5')

def train3(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,modelname):
    embedding_layer = Embedding(max_token + 1,
                                embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=False)
    print(modelname+'Build model...')
    model = Sequential()
    model.add(embedding_layer)
    # model.add(SimpleRNN(128, activation="relu"))
    # model.add(LSTM(128))
    # model.add(Bidirectional(LSTM(200))) ### 输出维度64 GRU
    model.add(Bidirectional(GRU(128)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    # lstm常选参数model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # a stateful LSTM model
    # lahead: the input sequence length that the LSTM
    # https://github.com/keras-team/keras/blob/master/examples/lstm_stateful.py
    # model = Sequential()
    # model.add(LSTM(20,input_shape=(lahead, 1),
    #               batch_size=batch_size,
    #               stateful=stateful))
    # model.add(Dense(1))
    # model.compile(loss='mse', optimizer='adam')

    # patience经过几个epoch后loss不在变化停止训练
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
    print('Train...')
    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=(x_test, y_test), callbacks=[early_stopping])
    # print(hist.history)
    ##输出loss与acc到日志文件
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=logpath, level=logging.DEBUG, format=log_format)
    logging.warning(modelname)
    for i in range(len(hist.history["acc"])):
        strlog=str(i+1)+" Epoch "+"-loss: "+str(hist.history["loss"][i])+" -acc: "+str(hist.history["acc"][i])+" -val_loss: "+str(hist.history["val_loss"][i])+" -val_acc: "+str(hist.history["val_acc"][i])
        logging.warning(strlog)

    model.save(modelpath + modelname + '.h5')