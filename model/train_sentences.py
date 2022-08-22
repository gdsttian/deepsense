# -*- coding: utf-8 -*-
'''
Decomposable attention model with features

Do not Distribute

training/validation data format:
    ["sentence","sentence_pmid","pmid","articleTitle","abstract","sentence_year","year","publicationType",
    "sum_citation","normalized_citation","journal_IF","","title_score","title_abstract_score","title_coverage",
    "abstract_coverage","","label"]
    
Test data format:
    [sentence","sentence_pmid","pmid","citation_pmid","articleTitle","abstract","sentence_year","year",
    "publicationType","sum_citation","normalized_citation"," journal_IF"," title_score","title_abstract_score",
    "title_coverage","abstract_coverage","rank"]
    
# Reference
[A Decomposable Attention Model for Natural Language Inference] (https://arxiv.org/abs/1606.01933)
'''
########################################
## import packages
########################################
import os
import tensorflow as tf
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from keras.models import Model,load_model
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from function_complete import *
from model_utils import opcount,plot_top_hist,create_submission,calculate_rank
from tokenizer_utils import load_tokenizer
# config = tf.ConfigProto()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True # Limit memory usage

os.environ["CUDA_VISIBLE_DEVICES"]="2" # Assign GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[2], 'GPU')

# sess = tf.Session(config=config)
sess = tf.compat.v1.Session(config=config)
########################################
## set directories and parameters
########################################

if_train=True 
conti_run=False
TEST_INPUT = '../case_SQL_testing_final_complete/' # Test data folder　
BASE_DIR = '../input/'
OUT_DIR = '../output/'
MODEL_dir='../model/'
EMBED_MAT_FILE = MODEL_dir + 'embedding_matrix.npy' #Word embedding file
TRAIN_DATA_FILE = BASE_DIR + 'train_sentences.tsv'
VALID_DATA_FILE = BASE_DIR + 'valid_sentences.tsv'

TOKENIZER_FILE = MODEL_dir + 'tokenizer_all_lower.pickle'

MAX_SENTENCE_LENGTH=100
MAX_ABSTRACT_LENGTH=1000
MAX_NB_WORDS = 1500000
EMBEDDING_DIM = 300
gpus=1
BATCH_SIZE=20*gpus
PRED_BATCH_SIZE=5
TRAIN_SIZE=2806978
VALID_SIZE=444397

mask=True
projection_dim = 300 
dense_dim = 300 
compare_dim = 500
projection_hidden = 0
compare_dropout = 0.2
projection_dropout = 0.2 
dense_dropout = 0.2 


STAMP = 'decomp_model_full_sentence'
last_model_path = MODEL_dir + STAMP + '_last.h5'
bst_model_path = MODEL_dir + STAMP + '.h5'
model_plot_path= MODEL_dir + STAMP + '.png'
########################################
## index word vectors
########################################
print('Indexing word vectors')
start = time.time()
# loading Tokenizer
tokenizer,nb_words=load_tokenizer(TOKENIZER_FILE,MAX_NB_WORDS)

cust=cust_function()
if if_train==True:
    if not os.path.exists(MODEL_dir):
        os.makedirs(MODEL_dir)

    
    ########################################
    ## prepare embeddings
    ########################################
 
    embedding_matrix=np.load(EMBED_MAT_FILE)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    
    gc.collect()
    
    if conti_run==True:

        model=load_model(last_model_path,custom_objects= cust)
        model.set_weights(model.get_weights())    
    ########################################
    ## define the model structure
    ########################################
    else:
        model=decomposable_attention_complete(embedding_matrix,nb_words,EMBEDDING_DIM,MAX_SENTENCE_LENGTH,MAX_ABSTRACT_LENGTH,
                                   projection_dim=projection_dim, projection_hidden=projection_hidden, projection_dropout=projection_dropout,
                                   compare_dim=compare_dim, compare_dropout=compare_dropout,
                                   dense_dim=dense_dim, dense_dropout=0.2,gpus=gpus,mask=mask)
        plot_model(model, to_file=model_plot_path, show_shapes=True)
    
    
    print(bst_model_path)
    
    print(model.summary())

########################################
## train the model
########################################
    
    class_weight = {0: 1, 1: 1}
    
    
    early_stopping =EarlyStopping(monitor='val_loss',mode='min', patience=15)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True)
    #tbCallBack = TensorBoard(log_dir='../Graph', histogram_freq=0, embeddings_freq=0,write_graph=True)
    de_hist=model.fit(data_generater_complete(TRAIN_DATA_FILE,BATCH_SIZE,tokenizer,
                                               MAX_SENTENCE_LENGTH,MAX_ABSTRACT_LENGTH),
                        steps_per_epoch=TRAIN_SIZE//BATCH_SIZE, epochs=20,
                        validation_data=data_generater_complete(VALID_DATA_FILE,BATCH_SIZE,tokenizer,
                                                       MAX_SENTENCE_LENGTH,MAX_ABSTRACT_LENGTH),
                        validation_steps=VALID_SIZE//BATCH_SIZE,shuffle=True,class_weight=class_weight, 
                       verbose=2, callbacks=[ model_checkpoint,early_stopping])
    model.save(last_model_path)
    #bst_val_score = max(de_hist.history['val_f1'])
    
    gc.collect()

