#%load_ext autoreload
#%aimport helper, tests
#%autoreload 1
#instead of the previouse command

from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 1')

import helper
import numpy as np
from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed, RepeatVector, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras import callbacks

#Verify access to the GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#Load Data   
preproc_source_sentences, preproc_target_sentences, source_tokenizer, target_tokenizer =\
helper.Pickle_out_data("Preprocessed_Data.pickle")
X_train, X_test, Y_train, Y_test=helper.Pickle_out_data("Split_Test_Train_Data.pickle")
    
max_source_sequence_length = preproc_source_sentences.shape[1]
max_target_sequence_length = preproc_target_sentences.shape[1]
source_vocab_size = len(source_tokenizer.word_index)+1
target_vocab_size = len(target_tokenizer.word_index)+1


# load the whole embedding into memory
embeddings_index = dict()
f = open('E:/Jalesiyan/Dataminig/Softwares/Embedding Models/glove.6B/glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((source_vocab_size, 100))
for word, i in source_tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
        
#e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)


# Define Model
def model_final(input_shape, output_sequence_length, s_size, t_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement

    # Hyperparameters
    learning_rate = 0.003
    
    # Build the layers    
    model = Sequential()
    # Embedding
    model.add(Embedding(s_size, 100, input_length=input_shape[1],
                         input_shape=input_shape[1:], weights=[embedding_matrix], trainable=False))
    # Encoder
    model.add(Bidirectional(GRU(100)))
    model.add(RepeatVector(output_sequence_length))
    # Decoder
    model.add(Bidirectional(GRU(100, return_sequences=True)))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(t_size, activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model

model = model_final(preproc_source_sentences.shape,preproc_target_sentences.shape[1],
                    len(source_tokenizer.word_index)+1,
                    len(target_tokenizer.word_index)+1)
model.summary()

#CallBacks
mfile = 'models/model-Glove.model.h5'
model_checkpoint=callbacks.ModelCheckpoint(mfile, monitor='accuracy', save_best_only=True, save_weights_only=True)
logger=callbacks.CSVLogger('results/Glove_training.log')
tensorboard=callbacks.TensorBoard(log_dir='results/Glove_tensprboard')
callbacks=[logger, tensorboard]

#Training model and save callbacks:
#model.fit(X_train, Y_train, batch_size=1024, epochs=25, validation_split=0.1, callbacks=callbacks)

#Training model and save callbacks:
model.fit(X_train, Y_train, batch_size=1024, epochs=1, validation_split=0.1)

Predicted_by_Glove = model.predict(X_test, len(X_test))
"""
#Save Model
helper.save_model(model, 'models/Glove')
helper.Pickle_in_Data(Predicted_by_Glove)
"""