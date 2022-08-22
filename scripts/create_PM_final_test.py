# -*- coding: utf-8 -*-
"""
Create PubMed test data
"""
import os
import time
import json
from text_to_wordlist import text_to_wordlist
import pandas as pd

year_max=2020

def load_json_citation(file):
    with open(file) as json_file:
        data = json.load(json_file)
    return data

def load_normalized_citation_by_year(BASE_DIR):
    normalized_citation_by_year={}
    with open(BASE_DIR+'normalized_citation_by_year.tsv') as f:
        head=next(f)
        for line in f:
            items=line.split('\t')
            year=int(items[0])
            citation_accumulate=int(items[1])
            avg=float(items[4][:-1])
            if normalized_citation_by_year.get(year):
                normalized_citation_by_year[year][citation_accumulate]=avg
            else:
                normalized_citation_by_year[year]={citation_accumulate:avg}
    return normalized_citation_by_year
def cal_sum_citation(sentence_year,pmid,citations_dict):
    sum_citation=0
    
    pmid=str(pmid)
    cited_years=citations_dict.get(pmid)
    if cited_years:
        for key in sorted(cited_years.keys()):
            cited_year=int(key)
            if int(sentence_year) >= cited_year:
                
                sum_citation+=cited_years[key]
        return sum_citation
    else:
        return 0
def cal_coverage(sentence,target):

    sentence_terms = sentence.split()
    target_terms = target.split()

    count = 0
    for item in sentence_terms:
        if item in target_terms:
            count += 1

    coverage = 0.0
    try:
        coverage = float(count) / float(len(sentence_terms))
        #print(coverage)
    except:
        pass

    return coverage

def create_PM_final_test(TEST_PM_INPUT,output,reader_abs_citation_IF,citations_dict,normalized_citation_by_year):
    if not os.path.exists(output):
        os.mkdir(output)
    indexs=set(reader_abs_citation_IF.index)
    for filename in os.listdir(TEST_PM_INPUT):
        
        #number = int(filename[len('DL_data_search_result_sentence_citation'):-4])
        number = int(filename.split('_')[-1].split('.')[0])
        if number >=0 and number <= 38:
            start=time.time()
            sentence_not_in_abstract_count=0
            print(filename)
            if os.path.exists(output + filename):
                continue
            f = open(TEST_PM_INPUT + filename, 'r',encoding='utf-8')
            f2 = open(output + filename , 'w',encoding='utf-8')
            f2.write('sentence')
            f2.write('\t')
            f2.write('sentence_pmid')
            f2.write('\t')
            f2.write('pmid')
            f2.write('\t')
            f2.write('citation_pmid')
            f2.write('\t')
            f2.write('articleTitle')
            f2.write('\t')
            f2.write('abstract')
            f2.write('\t')
            f2.write('sentence_year')
            f2.write('\t')
            f2.write('year')
            f2.write('\t')
            f2.write('publicationType')
            f2.write('\t')
            f2.write('sum_citation')
            f2.write('\t')
            f2.write('normalized_citation')
            f2.write('\t')
            f2.write('journal_IF')
            f2.write('\t')
            f2.write('title_score')
            f2.write('\t')
            f2.write('title_abstract_score')
            f2.write('\t')
            f2.write('title_coverage')
            f2.write('\t')
            f2.write('abstract_coverage')
            f2.write('\t')
            f2.write('rank')
            #f2.write('\t')
            #f2.write('sentence_orig')
            f2.write('\n')
            
            old_sen=''
            rank=1
            for line in f.readlines()[1:]:
                line = line.strip()
                #print(line[:20])
                items = line.split('\t')
                sentence = text_to_wordlist(items[0])
                sentence_pmid=int(items[1])
                if items[2]=="":
                    items[2]='99999'
                pmid = int(items[2])
                citation_pmid = int(items[3])
                #sentence_orig=items[4]

             
                title_score=0#No need socre
                title_abstract_score=0#No need socre
                if old_sen!=sentence:
                    rank=1
                
                old_sen=sentence
                if pmid in indexs:
                    ArticleTitle=reader_abs_citation_IF.loc[pmid]['articleTitle']
                    abstract=reader_abs_citation_IF.loc[pmid]['abstract']
                    ArticleTitle=text_to_wordlist(ArticleTitle)
                    abstract=text_to_wordlist(abstract)
                    title_coverage=cal_coverage(sentence,ArticleTitle)
                    abstract_coverage=cal_coverage(sentence,abstract)
                    if sentence_pmid in indexs:
                        sentence_year=int(reader_abs_citation_IF.loc[sentence_pmid]['year'])


                    else:
                        print(str(sentence_pmid)+" is not in abstracts")
                        sentence_not_in_abstract_count+=1
                        print("sentence_pmid_not_in_abstract_count:"+str(sentence_not_in_abstract_count))
                        i=sentence_pmid+1
                        while i<sentence_pmid+1000:
                            if i in indexs:
                                sentence_year=int(reader_abs_citation_IF.loc[i]['year'])
                                print("year %d, %d"%(i,sentence_year))
                                break
                            else:
                                pass
                            i+=1
                    year=int(reader_abs_citation_IF.loc[pmid]['year'])
                    year=min(year,year_max)
                    publicationType=reader_abs_citation_IF.loc[pmid]['publicationType']
                    #citation=float(reader_abs_citation_IF.loc[pmid]['citation'])
                    
                    year_diff=max(sentence_year-year,0)
                    avg_citation_all_abstracts=normalized_citation_by_year[year][year_diff] #avg citation for specific year and diff
                    sum_citation=cal_sum_citation(sentence_year,pmid,citations_dict) #abstract's citation till the sentence_year
                    normalized_citation=sum_citation-avg_citation_all_abstracts
                    journal_IF=float(reader_abs_citation_IF.loc[pmid]['journal_IF'])
                    f2.write(sentence)
                    f2.write('\t')
                    f2.write(str(sentence_pmid))
                    f2.write('\t')
                    f2.write(str(pmid))
                    f2.write('\t')
                    f2.write(str(citation_pmid))
                    f2.write('\t')
                    f2.write(ArticleTitle)
                    f2.write('\t')
                    f2.write(abstract)
                    f2.write('\t')
                    f2.write(str(sentence_year))
                    f2.write('\t')
                    f2.write(str(year))
                    f2.write('\t')
                    f2.write(publicationType)
                    f2.write('\t')
                    f2.write(str(sum_citation))
                    f2.write('\t')
                    f2.write(str(normalized_citation))
                    f2.write('\t')
                    f2.write(str(journal_IF))
                    f2.write('\t')
                    f2.write(str(title_score))
                    f2.write('\t')
                    f2.write(str(title_abstract_score))
                    f2.write('\t')
                    f2.write(str(title_coverage))
                    f2.write('\t')
                    f2.write(str(abstract_coverage))
                    f2.write('\t')
                    f2.write(str(rank))
                    #f2.write('\t')
                    #f2.write(sentence_orig)
                    f2.write('\n')       
                    rank+=1

            f2.close()
            f.close()
            end=time.time()
            print("time: %f"%(end-start))
            
if __name__ == "__main__":
    BASE_DIR = 'input/'
    
    # TEST_PM_INPUT = '../case_testing_final_PubMed_only_keyword/'
    TEST_PM_INPUT = '../test_pubmed_bm_search_returns/'
    
    # output = 'case_PubMed_testing_final_keywordOnly/'
    output = 'test_pubmed_bm_dataset/'
    
    citations_dict=load_json_citation(BASE_DIR+'abstract_citations_by_year.json')
    
    abs_citation_IF_input_file = BASE_DIR + 'title_abstracts_complete.tsv'
    
    print("read %s:"%abs_citation_IF_input_file,flush=True)
    
    reader_abs_citation_IF=pd.read_csv(abs_citation_IF_input_file, encoding='utf-8',sep='\t', 
                                       index_col='pmid')
    print(reader_abs_citation_IF.shape)
    reader_abs_citation_IF = reader_abs_citation_IF.loc[~reader_abs_citation_IF.index.duplicated(keep='first')]
    print(reader_abs_citation_IF.shape)
    
    normalized_citation_by_year=load_normalized_citation_by_year(BASE_DIR)
    
    create_PM_final_test(TEST_PM_INPUT,output,reader_abs_citation_IF,citations_dict,normalized_citation_by_year)
