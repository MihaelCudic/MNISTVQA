
# coding: utf-8

# In[1]:

"""
    Import functions
"""
import numpy as np
import h5py
import re

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Merge, Dropout, Flatten
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers.convolutional import Convolution2D, MaxPooling2D

import theano


# In[5]:

def formatText(question, answer):
    Xq = []
    Y = []
    for q, a in zip(question, answer):
        xq = [wordInd[w] for w in [x.strip() for x in re.split('(\W+)?', q) if x.strip()]]
        y = np.zeros(len(wordInd) + 1)  # let's not forget that index 0 is reserved
        y[wordInd[a]] = 1
        Xq.append(xq)
        Y.append(y)

    return pad_sequences(Xq), np.array(Y)


# In[6]:

# Create vocab variables
title = "VQAData.hdf5"
fin = h5py.File(title, "r")
vocab = fin['vocab'][:]
vocabSize = len(vocab) + 1
wordInd = dict((c, i + 1) for i, c in enumerate(vocab))

C_train = fin['C_train'][:]
Q_train = fin['Q_train'][:]
A_train = fin['A_train'][:]

C_valid = fin['C_valid'][:]
Q_valid = fin['Q_valid'][:]
A_valid = fin['A_valid'][:]

# Format data
Q_train, A_train = formatText(Q_train, A_train)
Q_valid, A_valid = formatText(Q_valid, A_valid)


# In[8]:

"""
    Generate C+R NN modules
"""
canDim = [64, 64]

# Canvas CNN
cnn = Sequential()

cnn.add(Convolution2D(8, 3, 3, input_shape=(1, canDim[0], canDim[1]), border_mode='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Convolution2D(128, 3, 3, activation='relu'))
cnn.add(Dropout(0.25))
cnn.add(Flatten())

# Question RNN
RNN = recurrent.LSTM
qrnn = Sequential()

qrnn.add(Embedding(vocabSize, 50, mask_zero=True))
qrnn.add(RNN(60, activation='relu', return_sequences=False))
qrnn.add(Dropout(0.3))

# Merged NN
model = Sequential()

model.add(Merge([cnn, qrnn], mode='concat', concat_axis=-1))
model.add(Dropout(0.3))
model.add(Dense(vocabSize, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')


# In[9]:

print('Training')

epoch = 40
batchSize = 50

model.fit([C_train, Q_train], A_train, batch_size=batchSize, nb_epoch=epoch, validation_data=([C_valid, Q_valid], A_valid), show_accuracy=True)
#loss, acc = model.evaluate([C_test, Q_test], A_test, show_accuracy=True, verbose=0)
#print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
del C_train, Q_train, A_train
del C_valid, Q_valid, A_valid


# In[10]:

del C_train, Q_train, A_train
del C_valid, Q_valid, A_valid

C_test = fin['C_test'][:]
Q_test = fin['Q_test'][:]
A_test = fin['A_test'][:]
T_test = fin['T_test'][:]
fin.close()

Q_test, A_test = formatText(Q_test, A_test)
#loss, acc = model.evaluate([C_test, Q_test], A_test, show_accuracy=True, verbose=0)
#print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

predicted = model.predict_classes([C_test, Q_test], batch_size=32)
expected = [list(i).index(1) for i in A_test]

nTypes = len(set(T_test))
nTest = len(C_test)
correct = [0]*nTypes
total = [0]*nTypes
match = [predicted[ind] == expected[ind] for ind in xrange(nTest)]

for i in xrange(nTest):
    correct[T_test[i]] += 1 if match[i] else 0
    total[T_test[i]] += 1
    
percent = [float(x)/y for x,y in zip(correct, total)]

print(percent)


# In[12]:

nTypes = len(set(T_test))
nTest = len(C_test)
correct = [0]*nTypes
total = [0]*nTypes
match = [predicted[ind] == expected[ind] for ind in xrange(nTest)]

for i in xrange(nTest):
    correct[T_test[i]] += 1 if match[i] else 0
    total[T_test[i]] += 1
    
percent = [float(x)/y for x,y in zip(correct, total)]

print(percent)


# In[13]:

print(float(sum(correct))/sum(total))


# In[ ]:



