# -*- coding: utf-8 -*-
"""
Functions for output
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
import json

output = 'test_results_sql_bm25/' # test output folder of SQL_BM25 search test data

if not os.path.exists(output):
    os.mkdir(output)

def opcount(fname):
    with open(fname,encoding='utf8') as f:
        for line_number, _ in enumerate(f, 1):
            pass
    return line_number
def plot_top_hist(submission,OUT_DIR,STAMP):
    cri1=submission['test_label']==1
    submission_1=submission[cri1]
    print('data with label=1 length:%d'%submission_1.shape[0])
    cri2=submission['PM_rank']!=10001
    submission_1=submission[cri1 & cri2]
    print('data with label=1 and in PubMed results length:%d'%submission_1.shape[0])
    plt.hist([submission_1['PM_rank'],submission_1['de_rank']],bins='auto',range=(0,22),label=['PM_rank', 'de_rank'])
    plt.legend(loc='upper right')
    plt.xlabel('rank')        
    plt.ylabel('count')
    plt.savefig(OUT_DIR+'rank_hist_'+STAMP+'.png')

def create_submission(TEST_DATA_FILE,preds,TEST_SIZE,OUT_DIR,STAMP): 
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    test_texts_1 = []
    test_texts_2 = []

    test_labels = []

    test_pmids = []
    test_PM_rank=[]
    i=0
    with open(TEST_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        for values in reader:
            test_texts_1.append(values[0])
            #test_texts_2.append(values[6])
            
            test_pmids.append(int(values[1]))
            cite=int(values[2])
            if cite==int(values[1]):
                label=1
            else:
                label=0
            test_labels.append(label)
            test_PM_rank.append(int(values[5]))
            
            i=i+1
                  
    print('Found %s texts in %s' % (len(test_texts_1),TEST_DATA_FILE))
        
    print(len(preds))
    print(len(test_pmids))
    preds=preds[:len(test_pmids)]
    submission = pd.DataFrame({'pmids':test_pmids,
                               'sentence':test_texts_1, #'abstract':test_texts_2,
                               'test_matched':preds.ravel(),'test_label':test_labels,
                               'PM_rank':test_PM_rank})
    submission=submission.sort_values(['sentence','test_matched'],ascending=False)
    submission['de_rank']=submission.groupby('sentence')['test_matched'].rank(ascending=False).astype(int)
    submission['group_size']=submission.groupby('sentence')['sentence'].transform("count").astype(int)
    submission=submission[['sentence','pmids',#'abstract',
                           'test_label',
                           'test_matched','PM_rank','de_rank','group_size']]
    submission['avg']=(submission['PM_rank']+submission['de_rank'])/2.0
    submission['avg_rank']=submission.groupby('sentence')['avg'].rank(ascending=True).astype(int)
    submission.to_csv(OUT_DIR+'pred_'+STAMP+'.csv', index=False,encoding='utf-8_sig')
    return submission,test_labels
def check_exist(test_file):
    if os.path.exists(output + test_file.split('/')[-1]):
        print(output + test_file.split('/')[-1] + ' exist')
        return 1
    else:
        return 0

def calculate_rank(test_file, preds):
    re_ranks = {}
    f = open(test_file, 'r')
    f.readline()
    for i, line in enumerate(f):
        line = line.strip()
        items = line.split('\t')
        #sentence = items[0]
        sentence = f"{items[0]}|{items[1]}|{items[3]}"
        result_pmid = items[2]
        citation_pmid = items[3]
        try:
            sql_rank = items[16]
        except:
            print("No sentence:")
            print(line)
            continue

        re_ranks[sentence] = re_ranks.get(sentence, {'indexes':[]})
        re_ranks[sentence]['indexes'].append(i)
        if result_pmid == citation_pmid:
            re_ranks[sentence]['cirank'] = sql_rank
            re_ranks[sentence]['cindex'] = i
        
    tups = []
    number_of_sentence = 0
    for k, v in re_ranks.items():
        if 'cindex' in v:
            number_of_sentence += 1
            tups.append([k, v['cirank'], v['indexes'][0], v['cindex'], v['indexes'][-1]])

    f2 = open(output + test_file.split('/')[-1], 'w')
    f2.write('sentence')
    f2.write('\t')
    f2.write('citation')
    f2.write('\t')
    f2.write('sql_rank')
    f2.write('\t')
    f2.write('dl_rank')
    f2.write('\n')
    for tup in tups:
        rank = get_rank(tup, preds)
        #print(tup)
        #print(rank)
        f2.write(tup[0])
        f2.write('\t')
        f2.write(str(tup[0].split('|')[-1]))
        f2.write('\t')
        f2.write(str(tup[1]))
        f2.write('\t')
        f2.write(str(rank))
        f2.write('\n')

    #print(f'number of sentences: {number_of_sentence}')
    f2.close()
    f.close()
    
def get_rank(tup, preds):
    start_index = tup[2]
    position_index = tup[3]
    end_index = tup[4]

    rank = 1

    i = start_index
    score = preds[position_index]
    while i <= end_index:
        if preds[i] > score:
            rank += 1
        i += 1
    return rank

if __name__ == '__main__':

    '''
    preds = []
    path = 'case_SQL_testing_final_complete/'
    test_file = path + 'DL_data_search_result_sentence_citation530.csv'
    calculate_rank(test_file, preds)
    
    '''

    tup = ['this is a sentence', '3', 3, 5, 7]
    preds = [30, 0.6, 0.9, 0.2, 1, 0.06, 0.06, 8, 20, 5, 7]

    print(tup)
    print(preds)
    print(get_rank(tup, preds))
