# -*- coding: utf-8 -*-
from keras.layers import Dense, Input, Flatten,Permute,Reshape
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.layers import merge
from keras.models import Model
from keras import backend as K
from keras.layers.core import Lambda,RepeatVector
import logging
from keras.callbacks import EarlyStopping

def train(x_train, y_train, x_test, y_test,maxlen,max_token,embedding_matrix,embedding_dims,batch_size,epochs,logpath,modelpath,modelname):
    embedding_layer = Embedding(max_token + 1,
                                embedding_dims,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=True)
    # LSTM步长
    TIME_STEPS = maxlen
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
    def attention_3d_block2(inputs, new_layer_name):
        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(inputs.shape[2])
        a = Permute((2, 1))(inputs)
        a = Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
        a = Dense(TIME_STEPS, activation='softmax')(a)
        if SINGLE_ATTENTION_VECTOR:
            a = Lambda(lambda x: K.mean(x, axis=1), name=new_layer_name + '_' + 'dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1), name=new_layer_name + '_''attention_vec')(a)
        output_attention_mul = merge([inputs, a_probs], name=new_layer_name + '_''attention_mul', mode='mul')
        return output_attention_mul

    # 单向LSTM之后加入Attention
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

    # 双向LSTM词encoder  输入是 词标签数组
    sentence_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    forward_rnn = LSTM(100, return_sequences=True)
    backward_rnn = LSTM(100, return_sequences=True, go_backwards=True)
    lstm_out_f_rnn = forward_rnn(embedded_sequences)
    attention_f_mul = attention_3d_block2(lstm_out_f_rnn, "forward")
    lstm_out_b_rnn = backward_rnn(embedded_sequences)
    attention_b_mul = attention_3d_block2(lstm_out_b_rnn, "backward")
    attention_mul = merge([attention_f_mul, attention_b_mul], mode='concat', concat_axis=-1)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(sentence_input, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # patience经过几个epoch后loss不在变化停止训练
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
    print('Train...')
    # history = model.fit([doc_x_train, left_x_train, right_x_train], y_train, epochs = 1)
    # loss = history.history["loss"][0]
    hist = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                     nb_epoch=epochs, batch_size=batch_size, callbacks=[early_stopping])
    # print(hist.history)
    ##输出loss与acc到日志文件
    log_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=logpath, level=logging.DEBUG, format=log_format)
    logging.warning(modelname)
    for i in range(len(hist.history["acc"])):
        strlog=str(i+1)+" Epoch "+"-loss: "+str(hist.history["loss"][i])+" -acc: "+str(hist.history["acc"][i])+" -val_loss: "+str(hist.history["val_loss"][i])+" -val_acc: "+str(hist.history["val_acc"][i])
        logging.warning(strlog)

    model.save(modelpath + modelname + '.h5')