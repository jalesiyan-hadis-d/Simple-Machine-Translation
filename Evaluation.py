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
#import nltk

#Verify access to the GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#Load Data   
preproc_source_sentences, preproc_target_sentences, source_tokenizer, target_tokenizer =\
helper.Pickle_out_data("Preprocessed_Data.pickle")
X_train, X_test, Y_train, Y_test=helper.Pickle_out_data("Split_Test_Train_Data.pickle")


#model=helper.load_model("models/Glove.json")

model=helper.load_model("models/Glove.json")
model.summary()

#Predicting Test data

Predicted_sentences = model.predict(X_test, len(X_test))

## DON'T EDIT ANYTHING BELOW THIS LINE
y_id_to_word = {value: key for key, value in target_tokenizer.word_index.items()}
y_id_to_word[0] = '<PAD>'
#Bleu

predict_list=[]
for lst in Predicted_sentences:
   predict_list.append((np.argmax(lst,1)).tolist())

Y_test=Y_test.tolist()



test_sentence=[]
predict_sentce=[]
size = len(Y_test)
for i in range (0,size):
   test_sentence.append(([y_id_to_word[np.max(x)] for x in Y_test[i]]))
   predict_sentce.append(([y_id_to_word[np.max(x)] for x in predict_list[i]]))

#nltk.translate.bleu_score.corpus_bleu(Y_test, predict_list)

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
#weights = (0.5,0.5,0,0)
chencherry = SmoothingFunction()
#weights=(0.5,0.5,0,0)
#compare elements
#print ("compare:", hypo)
score=[]
#print(('-')*40,'\n')
for i in range(0,size):
    ref=test_sentence[i]
    hypo=predict_sentce[i]
    score.append(sentence_bleu([ref], hypo, smoothing_function=chencherry.method4))
    #print (corpus[i], 'has a similarity rate of:\t', score, '%')
    
np.average(score)