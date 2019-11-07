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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, RepeatVector, Bidirectional, Dropout
from tensorflow.keras import layers as L
#from keras.layers.embeddings import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras import callbacks
#from tensorflow.keras import backend
#Gpu
#For Keras
#from keras.callbacks import ModelCheckpoint
#from keras.models import Model, load_model, save_model, Sequential
#from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
#from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
#from keras.optimizers import Adam
import tensorflow as tf
with tf.device("/gpu:0"):
    # Setup operations
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
#    session = tf.compat.v1.Session(config=config)

with tf.compat.v1.Session(config=config) as sess:
    # Run your code
    #Verify access to the GPU
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


    #Load Data   
    preproc_source_sentences, preproc_target_sentences, source_tokenizer, target_tokenizer =\
    helper.Pickle_out_data("Preprocessed_Data.pickle")
    X_train, X_test, Y_train, Y_test=helper.Pickle_out_data("Split_Test_Train_Data.pickle")
    embedding_matrix=helper.Pickle_out_data("embedded.pickle")   
    max_source_sequence_length = preproc_source_sentences.shape[1]
    max_target_sequence_length = preproc_target_sentences.shape[1]
    source_vocab_size = len(source_tokenizer.word_index)+1
    target_vocab_size = len(target_tokenizer.word_index)+1



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
        learning_rate = 0.005
    
        # Build the layers    
        model = Sequential()
        # Embedding
        model.add(L.Embedding(s_size, 100, input_length=input_shape[1],
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
    mfile = 'models/Glove_training_bach32.model.h5'
    model_checkpoint=callbacks.ModelCheckpoint(mfile, monitor='accuracy', save_best_only=True, save_weights_only=True)
    logger=callbacks.CSVLogger('results/training_bach_32.log')
    tensorboard=callbacks.TensorBoard(log_dir='results/training_bach_32')
    callbacks=[logger, tensorboard]

    #Training model and save callbacks:
    #model.fit(X_train, Y_train, batch_size=1024, epochs=25, validation_split=0.1, callbacks=callbacks)

    #Training model and save callbacks:
    model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_split=0.01)
    
    Predicted_by_Glove = model.predict(X_test, len(X_test))

    #Save Model
    helper.save_model(model, 'models/Glove_training_bach_32')

