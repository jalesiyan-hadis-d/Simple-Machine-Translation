#%load_ext autoreload
#%aimport helper, tests
#%autoreload 1
#instead of the previouse command

from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 1')

from keras.preprocessing.sequence import pad_sequences
import numpy as np
import helper
import modelling as MD
from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed, RepeatVector, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras import callbacks
from sklearn.model_selection import train_test_split

#Verify access to the GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Load English data
source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
english_sentences = helper.load_data(source_path)
french_sentences = helper.load_data(target_path)
print('Dataset Loaded')

  
preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
    helper.preprocess_and_save_data(english_sentences, french_sentences)
#Save Data
    

    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)+1
french_vocab_size = len(french_tokenizer.word_index)+1

# Define Model
def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
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
    model.add(Embedding(english_vocab_size, 128, input_length=input_shape[1],
                         input_shape=input_shape[1:]))
    # Encoder
    model.add(Bidirectional(GRU(128)))
    model.add(RepeatVector(output_sequence_length))
    # Decoder
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['categorical_accuracy'])
    return model

  
    
# TODO: Train neural network using model_final

model = model_final(preproc_english_sentences.shape,preproc_french_sentences.shape[1],
                    len(english_tokenizer.word_index)+1,
                    len(french_tokenizer.word_index)+1)
model.summary()

X_train, X_test, Y_train, Y_test = train_test_split(preproc_english_sentences, preproc_french_sentences, test_size=0.001, random_state=7)

logger=callbacks.CSVLogger('training.log')
tensorboard=callbacks.TensorBoard(log_dir='./tensprboard')
callbacks=[logger, tensorboard]

model.fit(X_train, Y_train, batch_size=1024, epochs=25, validation_split=0.1, callbacks=callbacks)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%,\n%s: %.2f%%" % (model.metrics_names[0], scores[0], model.metrics_names[1], scores[1]*100))

MD.save_model(model)

############################################################################
#Print Test
x=preproc_english_sentences
y=preproc_french_sentences
x_tk=english_tokenizer
y_tk=french_tokenizer


## DON'T EDIT ANYTHING BELOW THIS LINE
y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
y_id_to_word[0] = '<PAD>'

sentence1 = 'the july is chilly'
sentence2 = 'he saw a old yellow truck'
sentence1 = [x_tk.word_index[word] for word in sentence1.split()]
sentence1 = pad_sequences([sentence1], maxlen=x.shape[-1], padding='post')

sentence2 = [x_tk.word_index[word] for word in sentence2.split()]
sentence2 = pad_sequences([sentence2], maxlen=x.shape[-1], padding='post')
sentences = np.array([sentence1[0],sentence2[0]])
#sentences = np.array([sentence[0], x[0]])
predictions = model.predict(sentences, len(sentences))

print('Sample 1:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))


print('Sample 2:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
print('Sample 2:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
#print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))
