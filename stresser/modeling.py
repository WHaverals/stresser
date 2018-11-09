from keras.models import Model
from keras.optimizers import Adam
from keras.layers import *

def build_model(vectorizer, embed_dim, num_layers, recurrent_dim, lr):
    input_ = Input(shape=(vectorizer.max_len,), dtype='int32', name='syll')
    
    m = Embedding(input_dim=len(vectorizer.syll2idx),
                  output_dim=embed_dim,
                  mask_zero=True,
                  input_length=vectorizer.max_len)(input_)

    for i in range(num_layers):
        if i == 0:
            curr_input = m
        else:
            curr_input = curr_enc_out
        
        curr_enc_out = Bidirectional(LSTM(units=recurrent_dim,
                                          return_sequences=True,
                                          activation='tanh',
                                          name='enc_lstm_'+str(i + 1)),
                                     merge_mode='sum')(curr_input)

    dense = TimeDistributed(Dense(4), name='dense')(curr_enc_out)
    output_ = Activation('softmax', name='out')(dense)
    model = Model(inputs=input_, outputs=output_)
    optim = Adam(lr=lr)
    model.compile(optimizer=optim,
                  loss={'out': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    return model