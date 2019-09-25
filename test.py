# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:04:12 2019

@author: h.jalisian
"""
from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 1')

import numpy as np
import modelling as MD
from keras.preprocessing.sequence import pad_sequences
import helper

loaded_model = MD.load_model()
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'MSE'])
x, y, x_tk, y_tk=helper.load_preprocess()

###############################################################
"""
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
predictions = loaded_model.predict(sentences, len(sentences))

print('Sample 1:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))


print('Sample 2:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
print('Sample 2:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))
"""

############################################################################################################
#################################################################
#Giving New Sentence
Text1 = (input())
#sentence = 'the july is chilly'
sentence = [x_tk.word_index[word] for word in Text1.split()]
sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
## DON'T EDIT ANYTHING BELOW THIS LINE
y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
y_id_to_word[0] = '<PAD>'

#sentences = np.array([sentence[0]])
predictions = loaded_model.predict(sentence, len(sentence))

print('Sample 1:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))


"""
print("Enter/Paste your content. Ctrl-D ( windows ) to save it.")
contents = []
while True:
    try:
        line = input()
    except EOFError:
        break
    contents.append(line  ) 
    

x_tk.word_index['am']
"""