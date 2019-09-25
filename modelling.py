# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:57:40 2019

@author: h.jalisian
"""
from keras.models import model_from_json
from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 1')


#import helper
#import numpy as np
#import Functions as FN
#from keras.preprocessing.sequence import pad_sequences
#from keras.models import model_from_json
#from keras.layers import GRU, Dense, TimeDistributed, RepeatVector, Bidirectional, Dropout
#from keras.layers.embeddings import Embedding
#from keras.optimizers import Adam
#from keras.losses import sparse_categorical_crossentropy


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    
    
def load_model():
   
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model