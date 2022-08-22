# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 20:24:21 2019

@author: benbe
"""
import numpy as np
import pandas as pd
import csv
import pickle
import gc
from keras.preprocessing.text import Tokenizer
from gensim.models import FastText as fText
BASE_DIR = 'input/'
MODEL_DIR = 'model/'
MAX_NB_WORDS = 1500000
EMBEDDING_DIM = 300
EMBEDDING_FILE = MODEL_DIR + "f_model"
EMBED_MAT_FILE = MODEL_DIR + "embedding_matrix.npy"
TOKENIZER_FILE = MODEL_DIR + 'tokenizer_all_lower.pickle'
DATA_FILE = BASE_DIR + 'title_abstracts_raw_lower_for_sqldb.csv'
#EMBEDDING_model = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDING_FILE,binary=True)
def build_embedding_matrix(EMBEDDING_FILE,word_index,MAX_NB_WORDS,EMBEDDING_DIM,EMBED_MAT_FILE):
    #### Build embedding_matrix #####

    EMBEDDING_model = fText.load_fasttext_format(EMBEDDING_FILE)
    
    '''
    #fastText_wv = fText.load_fasttext_format("../input/f_model") 
    #fastText_wv.wv.most_similar("test")

    '''
    
    print('Preparing embedding matrix')
    
    nb_words = min(MAX_NB_WORDS, len(word_index))+1
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    
    for word, i in word_index.items():
        embedding_vector=0
        try:
            embedding_vector = EMBEDDING_model.wv.get_vector(word)
        except:
            try:
                embedding_vector = EMBEDDING_model.wv.get_vector(word.lower())
        
            except:
                try:
                    embedding_vector = EMBEDDING_model.wv.get_vector(word.upper())
                except:
                    try:
                        embedding_vector = EMBEDDING_model.wv.get_vector(word.title())
                    except:
                        print(word+' no embedding')
                    
        if embedding_vector is not 0 and i < nb_words:
    
            embedding_matrix[i] = embedding_vector
    embedding_matrix=embedding_matrix.astype(np.float32)
    np.save(EMBED_MAT_FILE,embedding_matrix,allow_pickle=False)
#    with open(EMBED_MAT_FILE, 'wb') as handle:
#        pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del EMBEDDING_model
    return embedding_matrix

def build_tokenizer(MAX_NB_WORDS,TOKENIZER_FILE,DATA_FILE): 
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='!"#$%&()*+,./:;<=>?@[\]^_`{|}~',lower=False)
    texts_1 = []
    
    i=0
    len_test=0
    with open(DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            i+=1
            texts_1.append(values[2])
            if i%100000==0:
                tokenizer.fit_on_texts(texts_1)
                len_test+=len(texts_1)
                del texts_1
                gc.collect()
                texts_1 = []
                print(str(i)+values[0],flush=True)
                
          
    
        tokenizer.fit_on_texts(texts_1)
        len_test+=len(texts_1)
        
    # saving
    with open(TOKENIZER_FILE, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer.word_index

def load_tokenizer(TOKENIZER_FILE,MAX_NB_WORDS):
    with open(TOKENIZER_FILE, 'rb') as handle:
        tokenizer = pickle.load(handle) 
    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))
    nb_words = min(MAX_NB_WORDS, len(word_index))+1
    return tokenizer,nb_words

if __name__ == '__main__':   
    word_index=build_tokenizer(MAX_NB_WORDS,TOKENIZER_FILE,DATA_FILE)
#    tokenizer,_=load_tokenizer(TOKENIZER_FILE,MAX_NB_WORDS)
#    word_index = tokenizer.word_index
    build_embedding_matrix(EMBEDDING_FILE,word_index,MAX_NB_WORDS,EMBEDDING_DIM,EMBED_MAT_FILE)