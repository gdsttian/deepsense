"""

training/validation data format:
    ["sentence","sentence_pmid","pmid","articleTitle","abstract","sentence_year","year","publicationType",
    "sum_citation","normalized_citation","journal_IF","","title_score","title_abstract_score","title_coverage",
    "abstract_coverage","","label"]
    
Test data format:
    [sentence","sentence_pmid","pmid","citation_pmid","articleTitle","abstract","sentence_year","year",
    "publicationType","sum_citation","normalized_citation"," journal_IF"," title_score","title_abstract_score",
    "title_coverage","abstract_coverage","rank"]
    
    
This model implementation is based on the following reference code:
https://www.kaggle.com/lamdang/dl-models
"""


import numpy as np
import pandas as pd
import csv
import codecs
import json
import tensorflow as tf 
from keras.layers import *
from keras.activations import softmax
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
import keras.backend as K
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from layers import Masking2D, Softmax2D, MaskedConv1D, MaskedGlobalAveragePooling1D

"""
https://github.com/CyberZHG/keras-self-attention
"""
from keras_self_attention import SeqSelfAttention
min_year=1997
max_year=2019
year_range=max_year-min_year+1
year_dict={}
for i,year in enumerate(range(min_year,max_year+1)):
    year_dict[year]=i

### List of important Publication Types
PubTypes=['journal article',
'research support, non-u.s. govt',
'review',
'comparative study',
'research support, u.s. govt, p.h.s.',
'english abstract',
'case reports',
'research support, n.i.h., extramural',
'research support, u.s. govt, non-p.h.s.',
'randomized controlled trial',
'clinical trial',
'multicenter study',
'evaluation studies',
'meta-analysis',
'validation studies',
'systematic review',
'historical article',
'controlled clinical trial',
'observational study',
'research support, n.i.h., intramural',
'comment',
'clinical trial, phase ii',
'letter',
'biography',
'editorial',
'clinical trial, phase i',
'published erratum',
'clinical trial, phase iii',
'video-audio media',
'practice guideline',
'portrait',
'Others']
PubTypes_len=len(PubTypes)
PubTypes_dict={}
for i,PubType in enumerate(PubTypes):
    PubTypes_dict[PubType]=i
def cust_function():
    cust={'f1': f1}
    cust['MaskedGlobalAveragePooling1D']=MaskedGlobalAveragePooling1D
    cust.update(SeqSelfAttention.get_custom_objects())
    return cust
def sort_rank(data,col,top_num):
    cri1=data['test_label']==1
    cri2=data['PM_rank']!=10001 # The cited article is not in search result. 
    cri=data[col]<=top_num
    data_1=data[cri1 & cri2 & cri]
    data_1_count=data_1.shape[0]
    print("%s top %d count:%d"%(col,top_num,data_1_count))
    return data_1_count

def Publication_type_to_catgorical(pubs):
    catgorical_pubs=[0]*PubTypes_len
    publicationType=pubs.split('|')
    for i,item in enumerate(publicationType):
        if item not in PubTypes:
            catgorical_pubs[PubTypes_dict['Others']]=1
        else:
            catgorical_pubs[PubTypes_dict[item]]=1
    return np.array(catgorical_pubs)
def data_generater_complete(DATA_FILE,batch_size,tokenizer,MAX_SENTENCE_LENGTH,MAX_ABSTRACT_LENGTH):
    texts_1 = [] 
    texts_2 = []
    texts_3 = []
    labels = []
    publicationTypes = []
    years = []
    citations = []
    IFs = []
    
    title_scores=[]
    title_abstract_scores=[]
    title_coverages=[]
    abstract_coverages=[]
    i=0
    
    while True:
        with open(DATA_FILE, encoding='utf-8') as f:
            
            head=next(f)
            for line in f:
                data=line.split('\t')
                #print(data)
                sentence=data[0]

                title=data[3]
                abstract=data[4]
                year=int(data[6])
                publicationType_label=Publication_type_to_catgorical(data[7])
                
                citation=float(data[9])
                journal_IF=float(data[10])
                #title_score=float(data[11])  
                #title_abstract_score=float(data[12])
                title_coverage=float(data[13])
                abstract_coverage=float(data[14])
                label=int(data[15][:-1])

                texts_1.append(sentence)
                texts_2.append(title)
                texts_3.append(abstract)
                labels.append(label)
                
                year=min(max_year,year)
                year=max(min_year,year)
                year_label=year_dict[year]
                years.append(year_label)
                
                publicationTypes.append(publicationType_label)
                citations.append(citation)
                IFs.append(journal_IF)
     
                title_coverages.append(title_coverage)
                abstract_coverages.append(abstract_coverage)
                
                i=i+1
                if i>=batch_size:
           
                    sequences_1 = tokenizer.texts_to_sequences(texts_1)
                    sequences_2 = tokenizer.texts_to_sequences(texts_2)
                    sequences_3 = tokenizer.texts_to_sequences(texts_3)
                    data_1 = pad_sequences(sequences_1, truncating='post', maxlen=MAX_SENTENCE_LENGTH)
                    data_2 = pad_sequences(sequences_2, truncating='post', maxlen=MAX_SENTENCE_LENGTH)
                    data_3 = pad_sequences(sequences_3, truncating='post', maxlen=MAX_ABSTRACT_LENGTH)
                    
                    years=to_categorical(years,year_range)
                    publicationTypes=np.array(publicationTypes)
                    citations=np.array(citations)
                    IFs=np.array(IFs)
       
                    title_coverages=np.array(title_coverages)
                    abstract_coverages=np.array(abstract_coverages)
                    labels = np.array(labels)    

                    yield ([data_1,data_2,data_3,years,citations,IFs,publicationTypes,title_coverages,abstract_coverages],labels)
                    i=0
                    texts_1=[];texts_2=[];texts_3=[];publicationTypes=[];years=[];citations=[];IFs=[];labels=[]
                    #title_scores=[]
                    #title_abstract_scores=[]
                    title_coverages=[]
                    abstract_coverages=[]
def data_generater_test_complete(DATA_FILE,batch_size,tokenizer,MAX_SENTENCE_LENGTH,MAX_ABSTRACT_LENGTH):
        
    texts_1 = [] 
    texts_2 = []
    texts_3 = []
    labels = []
    publicationTypes = []
    years = []
    citations = []
    IFs = []
    title_scores=[]
    title_abstract_scores=[]
    title_coverages=[]
    abstract_coverages=[]
    i=0
    

    while True:
        with codecs.open(DATA_FILE, encoding='utf-8') as f:
            
            header = next(f)
            for line in f:
                data=line.split('\t')
                sentence=data[0]

                title=data[4] 
                abstract=data[5]
                year=int(data[7])
                publicationType_label=Publication_type_to_catgorical(data[8])
                
                citation=float(data[10])
                journal_IF=float(data[11])
                #title_score=float(data[12])  
                #title_abstract_score=float(data[13])
                title_coverage=float(data[14])
                abstract_coverage=float(data[15])
                
                    
                texts_1.append(sentence)
                texts_2.append(title)
                texts_3.append(abstract)
                
                
                year=min(max_year,year)
                year=max(min_year,year)
                year_label=year_dict[year]
                years.append(year_label)
                publicationTypes.append(publicationType_label)
                citations.append(citation)
                IFs.append(journal_IF)

                title_coverages.append(title_coverage)
                abstract_coverages.append(abstract_coverage)
                i=i+1
                if i>=batch_size:
                    sequences_1 = tokenizer.texts_to_sequences(texts_1)
                    sequences_2 = tokenizer.texts_to_sequences(texts_2)
                    sequences_3 = tokenizer.texts_to_sequences(texts_3)
                    data_1 = pad_sequences(sequences_1, truncating='post', maxlen=MAX_SENTENCE_LENGTH)
                    data_2 = pad_sequences(sequences_2, truncating='post', maxlen=MAX_SENTENCE_LENGTH)
                    data_3 = pad_sequences(sequences_3, truncating='post', maxlen=MAX_ABSTRACT_LENGTH)
                    
                    years=to_categorical(years,year_range)
                    publicationTypes=np.array(publicationTypes)
                    citations=np.array(citations)
                    IFs=np.array(IFs)

                    title_coverages=np.array(title_coverages)
                    abstract_coverages=np.array(abstract_coverages)
                    

                    yield ([data_1,data_2,data_3,years,citations,IFs,publicationTypes,title_coverages,abstract_coverages])
                    i=0
                    texts_1=[];texts_2=[];texts_3=[];publicationTypes=[];years=[];citations=[];IFs=[];labels=[]
                    #title_scores=[]
                    #title_abstract_scores=[]
                    title_coverages=[]
                    abstract_coverages=[]
                    
            # Last samples
            if len(texts_1) > 0 and len(texts_2) > 0 and len(texts_3) > 0:
                sequences_1 = tokenizer.texts_to_sequences(texts_1)
                sequences_2 = tokenizer.texts_to_sequences(texts_2)
                sequences_3 = tokenizer.texts_to_sequences(texts_3)
                data_1 = pad_sequences(sequences_1, truncating='post', maxlen=MAX_SENTENCE_LENGTH)
                data_2 = pad_sequences(sequences_2, truncating='post', maxlen=MAX_SENTENCE_LENGTH)
                data_3 = pad_sequences(sequences_3, truncating='post', maxlen=MAX_ABSTRACT_LENGTH)
            
                years=to_categorical(years,year_range)
                publicationTypes=np.array(publicationTypes)
                citations=np.array(citations)
                IFs=np.array(IFs)

                title_coverages=np.array(title_coverages)
                abstract_coverages=np.array(abstract_coverages)

                yield ([data_1,data_2,data_3,years,citations,IFs,publicationTypes,title_coverages,abstract_coverages])
        

def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape

def subtract(input_1, input_2):
    "Subtract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = subtract(input_1, input_2)
    out_= Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_) #Apply layer_ on each word vector separately
    out_ = node_
    return out_
def soft_max(x, axis):
    import tensorflow as tf
    return tf.nn.softmax(x, axis=axis)

def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2]) # Dot on same len, = input_1[k] dot input_2[k].T
    
    w_att_1 = Lambda(soft_max,arguments={'axis': 1},name='soft_max_1',
                     output_shape=unchanged_shape)(attention) # np.sum(w_att_1,axis=1)=1
    # ex : attention = [[[0.32       0.32       0.5       ]
    #                  [0.26       0.26       0.41000003]]]
    #      w_att_1 = [[[0.5149955  0.5149955  0.52248484]
    #                [0.4850045  0.4850045  0.4775152 ]]]
    
    # Permute=transport
    w_att_2 = Permute((2,1))(Lambda(soft_max,arguments={'axis': 2},name='soft_max_2',
                             output_shape=unchanged_shape)(attention)) # np.sum(w_att_2,axis=1)=1
    # ex : attention = [[[0.32       0.32       0.5       ]
    #                  [0.26       0.26       0.41000003]]]        
    #      w_att_2 = [[[0.31277198 0.3162721 ]
    #                [0.31277198 0.3162721 ]
    #                [0.37445605 0.36745578]]]
    
    in1_aligned = Dot(axes=1)([w_att_1, input_1]) # alpha = w_att_1[k].T dot input_1[k]
    in2_aligned = Dot(axes=1)([w_att_2, input_2]) # beta = w_att_2[k].T dot input_2[k]
    return in1_aligned, in2_aligned
def soft_attention_alignment2(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2]) # Dot on same len, = input_1[k] dot input_2[k].T
    
    w_att_1 = Lambda(soft_max,arguments={'axis': 1},name='soft_max_3',
                     output_shape=unchanged_shape)(attention) # np.sum(w_att_1,axis=1)=1
    # ex : attention = [[[0.32       0.32       0.5       ]
    #                  [0.26       0.26       0.41000003]]]
    #      w_att_1 = [[[0.5149955  0.5149955  0.52248484]
    #                [0.4850045  0.4850045  0.4775152 ]]]
    
    # Permute=transport
    w_att_2 = Permute((2,1))(Lambda(soft_max,arguments={'axis': 2},name='soft_max_4',
                             output_shape=unchanged_shape)(attention)) # np.sum(w_att_2,axis=1)=1
    # ex : attention = [[[0.32       0.32       0.5       ]
    #                  [0.26       0.26       0.41000003]]]        
    #      w_att_2 = [[[0.31277198 0.3162721 ]
    #                [0.31277198 0.3162721 ]
    #                [0.37445605 0.36745578]]]
    
    in1_aligned = Dot(axes=1)([w_att_1, input_1]) # alpha = w_att_1[k].T dot input_1[k]
    in2_aligned = Dot(axes=1)([w_att_2, input_2]) # beta = w_att_2[k].T dot input_2[k]
    return in1_aligned, in2_aligned
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
    

def decomposable_attention_complete(embedding_matrix, nb_words,EMBEDDING_DIM,MAX_SENTENCE_LENGTH,MAX_ABSTRACT_LENGTH,
                           projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=500, compare_dropout=0.2,
                           dense_dim=300, dense_dropout=0.2,
                           lr=1e-3, activation='elu', gpus=1,mask=True):
    # Based on: https://arxiv.org/abs/1606.01933
    maxlen1=MAX_SENTENCE_LENGTH
    maxlen2=MAX_SENTENCE_LENGTH
    maxlen3=MAX_ABSTRACT_LENGTH
    q1 = Input(name='q1',shape=(maxlen1,))
    q2 = Input(name='q2',shape=(maxlen2,))
    q3 = Input(name='q3',shape=(maxlen3,))
    q4 = Input(name='q4',shape=(year_range,))
    q5 = Input(name='q5',shape=(1,))
    q6 = Input(name='q6',shape=(1,))
    q7 = Input(name='q7',shape=(PubTypes_len,))
#    q8 = Input(name='q8',shape=(1,))
#    q9 = Input(name='q9',shape=(1,))
    q10 = Input(name='q10',shape=(1,))
    q11 = Input(name='q11',shape=(1,))
    
    # Embedding
    embedding = Embedding(nb_words,
        EMBEDDING_DIM,
        mask_zero=mask,
        weights=[embedding_matrix],
        trainable=False
        )
    
    q1_embed = embedding(q1)
    
    q2_embed = embedding(q2)
    
    q3_embed = embedding(q3)
    
    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
                Dense(projection_dim, activation=activation),
                Dropout(rate=projection_dropout),
            ])
    projection_layers.extend([
            Dense(projection_dim, activation=activation),
            Dropout(rate=projection_dropout),
        ])
    q1_encoded = time_distributed(q1_embed, projection_layers)
    q2_encoded = time_distributed(q2_embed, projection_layers)
    q3_encoded = time_distributed(q3_embed, projection_layers)
    
    # Attention
    q12_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    q13_aligned, q3_aligned = soft_attention_alignment2(q1_encoded, q3_encoded)    
    
    # Compare
    q12_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q12_aligned, submult(q2_encoded, q12_aligned)]) 

    q13_combined = Concatenate()([q1_encoded, q3_aligned, submult(q1_encoded, q3_aligned)])
    q3_combined = Concatenate()([q3_encoded, q13_aligned, submult(q3_encoded, q13_aligned)])
    
    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    q12_compare = time_distributed(q12_combined, compare_layers)
    q2_compare = time_distributed(q2_combined, compare_layers)
  
    q13_compare = time_distributed(q13_combined, compare_layers)
    q3_compare = time_distributed(q3_combined, compare_layers)
    # Aggregate
    q12_rep = MaskedGlobalAveragePooling1D()(Masking()(q12_compare))
    q2_rep = MaskedGlobalAveragePooling1D()(Masking()(q2_compare)) 
    q13_rep = MaskedGlobalAveragePooling1D()(Masking()(q13_compare))
    q3_rep = MaskedGlobalAveragePooling1D()(Masking()(q3_compare))
#    print(type(q1_rep))
#    print(type(q2_rep))
#    print(type(q3))
    
    # Add features
    feature_merged = Concatenate()([q4,q5,q6,q7,q10,q11])
    feature_dense = Dense(32, activation=activation)(feature_merged)
    feature_dense = Dropout(dense_dropout)(feature_dense)
    feature_dense = BatchNormalization()(feature_dense)
    feature_dense = Dense(16, activation=activation)(feature_dense)
    feature_dense = Dropout(dense_dropout)(feature_dense)
    feature_dense = BatchNormalization()(feature_dense)    
    # Classifier
    merged = Concatenate()([q12_rep, q2_rep, q13_rep, q3_rep, feature_dense])
    
    dense = Dense(dense_dim, activation=activation)(merged)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    out_ = Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs=[q1, q2, q3, q4,q5,q6,q7,q10,q11], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', 
                      metrics=['binary_crossentropy','accuracy'])
    if gpus>=2:
        parallel_model = ModelMGPU(model, gpus=gpus)
    else:
        parallel_model=model
    parallel_model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', 
                      metrics=[f1,'accuracy'])

    return parallel_model


def plot_auc(fpr_keras, tpr_keras,auc_keras,label,path):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label=label+' (area = {:.3f})'.format(auc_keras))
    #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(path)
def plot_con_matrix(confmat,path):
   
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
    plt.xlabel('predicted label')        
    plt.ylabel('true label')
    plt.savefig(path)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
