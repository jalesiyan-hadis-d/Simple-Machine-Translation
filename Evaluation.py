# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:04:12 2019
"""
from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 1')

import helper
import numpy as np
import nltk

#Verify access to the GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#Load Data   
preproc_source_sentences, preproc_target_sentences, source_tokenizer, target_tokenizer =\
helper.Pickle_out_data("Preprocessed_Data.pickle")
X_train, X_test, Y_train, Y_test=helper.Pickle_out_data("Split_Test_Train_Data.pickle")


#model=helper.load_model("models/Glove.json")

model=helper.load_model("models/Glove-LSTM.json")
model.summary()

#Predicting Test data

Predicted_sentences = model.predict(X_test, len(X_test))



#Bleu

predict_list=[]
for lst in Predicted_sentences:
   predict_list.append((np.argmax(lst,1)).tolist())

Y_test=Y_test.tolist()

nltk.translate.bleu_score.corpus_bleu(Y_test, predict_list)



## DON'T EDIT ANYTHING BELOW THIS LINE
y_id_to_word = {value: key for key, value in target_tokenizer.word_index.items()}
y_id_to_word[0] = '<PAD>'
"""
for i in range (1,10):
    print('source %f:', i)
    print(' '.join([y_id_to_word[np.max(x)] for x in Y_test[i]]))
    
    print('Translate %f:',i)
    print(' '.join([y_id_to_word[np.max(x)] for x in predict_list[i]]))

"""