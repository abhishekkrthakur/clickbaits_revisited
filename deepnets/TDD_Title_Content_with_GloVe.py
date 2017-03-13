
# coding: utf-8
"""
Time distributed dense with GloVe embeddings (title + content text)
@author: Abhishek Thakur
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.engine.topology import Merge
from keras.layers import TimeDistributed, Lambda
from keras.callbacks import ModelCheckpoint
from keras import backend as K
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

tk.fit_on_texts(list(train.link_name.values) + list(train.textdata.values) + list(test.link_name.values) + list(test.textdata.values))
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

embeddings_index = {}
f = open('../data/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=80,
                     trainable=False))
model1.add(TimeDistributed(Dense(300, activation='relu')))
model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=80,
                     trainable=False))
model2.add(TimeDistributed(Dense(300, activation='relu')))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))


merged_model = Sequential()
merged_model.add(Merge([model1, model2], mode='concat'))
merged_model.add(BatchNormalization())

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
checkpoint = ModelCheckpoint('../data/weights_title+content_tdd.h5', monitor='val_acc', save_best_only=True, verbose=2)

merged_model.fit([x_train_title, x_train_textdata], y=ytrain_enc, 
                 batch_size=128, nb_epoch=200, verbose=2, validation_split=0.1, 
                 shuffle=True, callbacks=[checkpoint])
