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
from model_utils_pubmed_bm import opcount,plot_top_hist,create_submission,calculate_rank
from tokenizer_utils import load_tokenizer
# config = tf.ConfigProto()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True # Limit memory usage

os.environ["CUDA_VISIBLE_DEVICES"]="3" # Assign GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

# sess = tf.Session(config=config)
sess = tf.compat.v1.Session(config=config)
########################################
## set directories and parameters
########################################

if_train=True 
conti_run=False
TEST_INPUT = '../test_dataset_pubmed_bm/' # Test data folder of pubmed BM search
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


STAMP = 'decomp_model_full_sentence_new'
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

########################################
## Test the model
########################################
model=load_model(bst_model_path,custom_objects= cust)

for filename in os.listdir(TEST_INPUT):
    TEST_DATA_FILE = TEST_INPUT + filename
    TEST_SIZE= len(open(TEST_DATA_FILE).readlines(  )) - 1
    print(filename)
    print('test data size:%d'%TEST_SIZE)


    ########################################
    ## make the submission
    ########################################
    print('Start making the submission',flush=True)
    preds = model.predict(data_generater_test_complete(
            TEST_DATA_FILE,
            PRED_BATCH_SIZE,tokenizer,MAX_SENTENCE_LENGTH,MAX_ABSTRACT_LENGTH),max_queue_size=5,
                    steps=TEST_SIZE//PRED_BATCH_SIZE + 1, verbose=2)

    calculate_rank(TEST_DATA_FILE, preds)

    print(f'Length of prediction: {len(preds)}')

    f = open(TEST_DATA_FILE, 'r')
    lines = f.readlines()[1:]
    print(f'Length of testing set: {len(lines)}')
    f.close()

    ''' Full output if necessary. Huge file.
    submission = pd.read_csv(TEST_DATA_FILE,sep='\t',encoding='utf-8')
    submission['model_pred']=preds.ravel()[:TEST_SIZE]
    submission['label']=submission['citation_pmid']==submission['pmid']
    submission['de_rank']=submission.groupby('sentence')['model_pred'].rank(ascending=False).astype(int)
    submission.to_csv(OUT_DIR+'pred_'+filename, index=False,encoding='utf-8_sig')
    '''
    
