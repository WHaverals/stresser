from keras.models import Model
from keras.optimizers import Adam
from keras.layers import *

from keras_contrib.layers import CRF

def build_model(vectorizer, embed_dim, num_layers, recurrent_dim,
                lr, dropout, no_crf=False, num_classes=3):
    input_ = Input(shape=(vectorizer.max_len,), dtype='int32')

    m = Embedding(input_dim=len(vectorizer.syll2idx),
                  output_dim=embed_dim,
                  mask_zero=True,
                  input_length=vectorizer.max_len)(input_)
    m = Dropout(dropout)(m)

    for i in range(num_layers):
        if i == 0:
            curr_input = m
        else:
            curr_input = curr_enc_out
        
        curr_enc_out = Bidirectional(LSTM(units=recurrent_dim,
                                          return_sequences=True,
                                          activation='tanh',
                                          recurrent_dropout=dropout,
                                          name='enc_lstm_'+str(i + 1)),
                                     merge_mode='sum')(curr_input)
    curr_enc_out = Dropout(dropout)(curr_enc_out)
    dense = TimeDistributed(Dense(num_classes, activation='relu'), name='dense')(curr_enc_out)
    optim = Adam(lr=lr)

    if not no_crf:
        crf = CRF(num_classes)
        output_ = crf(dense)
        model = Model(inputs=input_, outputs=output_)
        model.compile(optimizer=optim, loss=crf.loss_function, metrics=[crf.accuracy])
    else:
        output_ = Activation('softmax', name='out')(dense)
        model = Model(inputs=input_, outputs=output_)
        model.compile(optimizer=optim,
                      loss={'out': 'categorical_crossentropy'},
                      metrics=['accuracy'])
    return model