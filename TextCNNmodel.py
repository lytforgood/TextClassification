# -*- coding: utf-8 -*-
from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout

from keras.callbacks import EarlyStopping
import logging


def train(x_train, y_train, x_test, y_test, maxlen, max_token, embedding_matrix, embedding_dims, batch_size, epochs,
          logpath, modelpath, modelname):
    print(modelname + 'Build model...')
    sentence = Input((maxlen,))
    embedding_layer = Embedding(max_token + 1,
                                embedding_dims,
                                input_length=maxlen,
                                weights=[embedding_matrix])
    sentence_embedding = embedding_layer(sentence)
    c2 = Conv1D(128, 3, activation='relu')(sentence_embedding)
    p2 = GlobalMaxPooling1D()(c2)

    c3 = Conv1D(128, 4, activation='relu')(sentence_embedding)
    p3 = GlobalMaxPooling1D()(c3)

    c4 = Conv1D(128, 5, activation='relu')(sentence_embedding)
    p4 = GlobalMaxPooling1D()(c4)

    x = Concatenate()([p2, p3, p4])
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=sentence, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # print(model.summary())
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
        strlog = str(i + 1) + " Epoch " + "-loss: " + str(hist.history["loss"][i]) + " -acc: " + str(
            hist.history["acc"][i]) + " -val_loss: " + str(hist.history["val_loss"][i]) + " -val_acc: " + str(
            hist.history["val_acc"][i])
        logging.warning(strlog)

    model.save(modelpath + modelname + '.h5')


if __name__ == '__main__':
    print('11')
