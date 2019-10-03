# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 07:31:24 2019

@author: h.jalisian
"""

from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 1')

import helper
import os
from sklearn.model_selection import train_test_split


"""
def maybe_download(filename, work_directory):
 #   Download the data from Yann's website, unless it's already here.
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath
"""
def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')

# Load data
source_path = ' data/vocab_en'
filename='http://yann.lecun.com/exdb/mnist/'
target_path = 'data/vocab_fr'
source_sentences = load_data(source_path)
target_sentences = load_data(target_path)
print('Dataset Loaded')

#Preprocessing
preproc_resource_sentences, preproc_target_sentences, resource_tokenizer, target_tokenizer =\
   helper.preprocess_data(source_sentences, target_sentences)

# Save Data
pickle_name="Preprocessed_Data"
helper.Pickle_in_data(preproc_resource_sentences, preproc_target_sentences, resource_tokenizer, target_tokenizer,pickle_name)
X_train, X_test, Y_train, Y_test = train_test_split(preproc_resource_sentences, preproc_target_sentences, test_size=0.001, random_state=7)
pickle_name="Split_Test_Train_Data"
helper.Pickle_in_data(X_train, X_test, Y_train, Y_test,pickle_name)