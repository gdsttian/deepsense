# -*- coding: utf-8 -*-
"""
Search PubMed sentences
"""

import re
import random

import pandas as pd
import multiprocessing as mp
import time

from text_to_wordlist import text_to_wordlist
from PM_function import create_data_PubMed

import os
import csv


from create_PM_final_test import create_PM_final_test
BASE_DIR = 'input/'

TEST_PM_INPUT = '../case_testing_final_PubMed_only_keyword/'
#TEST_sentence_INPUT = '../case_DL_testing_final_with_score_coverage/'
TEST_sentence_INPUT = '../case_DL_testing_final_only_keyword_with_all_score_coverage/'

abs_citation_IF_input_file = BASE_DIR + 'title_abstracts_complete_only_features.tsv'

retmax=1000
START_NO=463

if __name__ == "__main__":
    
    start = time.time()

    if not os.path.exists(TEST_PM_INPUT):
        os.mkdir(TEST_PM_INPUT)

    reader_abs_citation_IF=pd.read_csv(abs_citation_IF_input_file, encoding='utf-8',sep='\t',
                                   dtype={'year': int},
                                   usecols=['pmid','year'],index_col='pmid')
    print(reader_abs_citation_IF.shape)
    reader_abs_citation_IF = reader_abs_citation_IF.loc[~reader_abs_citation_IF.index.duplicated(keep='first')]
    print(reader_abs_citation_IF.shape) 
    indexs=set(reader_abs_citation_IF.index)   
    print("Pubmed search",flush=True)  
    count=0
    for filename in sorted(os.listdir(TEST_sentence_INPUT)):
        
        sentence_not_in_abstract_count=0
        print("%d, %s"%(count,filename))
        count+=1
        post=filename.split('citation')[1]
        if int(post[:3]) < START_NO:
            continue
        
        TEST_OUT_FILE = TEST_PM_INPUT + filename
        TEST_TEMP_FILE='sen_pmids'+post
        TEST_sentence_FILE = TEST_sentence_INPUT + filename

       
        print(filename,flush=True)
     
        sen_pmids=set()
        
        with open(TEST_sentence_FILE, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            for values in reader:
                sen=text_to_wordlist(values[0])
                sen_pmid=int(values[1])
                try:
                    cite_pmid=int(values[2])
                except:
                    print("No sentence:")
                    print(values)
                    continue
                
                sentence_orig=values[3]
                if sen_pmid in indexs:
                    year=int(reader_abs_citation_IF.loc[sen_pmid]['year'])
               
                else:
                    sentence_not_in_abstract_count+=1
                    print("sentence_pmid_not_in_abstract_count:"+str(sentence_not_in_abstract_count))
                    i=sen_pmid+1
                    while i<sen_pmid+1000:
                        if i in indexs:
                            year=int(reader_abs_citation_IF.loc[i]['year'])
                            print("year %d, %d"%(i,year))
                            break
                        else:
                            pass
                        i+=1
                    
                    if (sen,sen_pmid,cite_pmid,year,sentence_orig) not in sen_pmids:
                        print(str(sen_pmid)+' is not in database',flush=True)
                        
                sen_pmids.add((sen,sen_pmid,cite_pmid,year,sentence_orig))

            
        create_data_PubMed(sen_pmids,TEST_OUT_FILE,
                                    'log_pubmed_search_review.txt', 

                                    op='OR')
    
#
    print("finish")

    end = time.time()
    print('time:'+ str(end - start)) 
    