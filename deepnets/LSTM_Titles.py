# coding: utf-8
"""
Simple LSTM only on Titles
@author: Abhishek Thakur
"""

import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

y_train = train.label.values
y_test = test.label.values

tk = text.Tokenizer(nb_words=200000)
train.link_name = train.link_name.astype(str)
test.link_name = test.link_name.astype(str)
train.textdata = train.textdata.astype(str)
test.textdata = test.textdata.astype(str)

max_len = 80

tk.fit_on_texts(list(train.link_name.values) + list(train.textdata.values) + list(test.link_name.values) + list(
    test.textdata.values))
x_train_title = tk.texts_to_sequences(train.link_name.values)
x_train_title = sequence.pad_sequences(x_train_title, maxlen=max_len)

x_train_textdata = tk.texts_to_sequences(train.textdata.values)
x_train_textdata = sequence.pad_sequences(x_train_textdata, maxlen=max_len)

x_test_title = tk.texts_to_sequences(test.link_name.values)
x_test_title = sequence.pad_sequences(x_test_title, maxlen=max_len)

x_test_textdata = tk.texts_to_sequences(test.textdata.values)
x_test_textdata = sequence.pad_sequences(x_test_textdata, maxlen=max_len)

word_index = tk.word_index
ytrain_enc = np_utils.to_categorical(y_train)

merged_model = Sequential()
merged_model.add(Embedding(len(word_index) + 1, 300, input_length=80, dropout=0.2))
merged_model.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1, 300, input_length=80, dropout=0.2))
model2.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

merged_model.add(Dense(200))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(200))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(2))
merged_model.add(Activation('softmax'))

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall'])

checkpoint = ModelCheckpoint('../data/weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

merged_model.fit(x_train_title, y=ytrain_enc,
                 batch_size=128, nb_epoch=200, verbose=2, validation_split=0.1,
                 shuffle=True, callbacks=[checkpoint])
