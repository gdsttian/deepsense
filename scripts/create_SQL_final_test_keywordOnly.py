'''
Create SQL test data
'''
import os
import time
import json
from text_to_wordlist import text_to_wordlist
import pandas as pd
#path = 'case_PM_testing_pmids/'

#output = 'case_PM_testing_final/'
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
def create_SQL_final_test(TEST_SQL_INPUT,output,reader_abs_citation_IF,citations_dict,normalized_citation_by_year):
    if not os.path.exists(output):
        os.mkdir(output)
    indexs=set(reader_abs_citation_IF.index)
    for filename in os.listdir(TEST_SQL_INPUT):
        
        number = int(filename[len('DL_data_search_result_sentence_citation'):-4])
        if number >510 and number <= 560:
            start=time.time()
            sentence_not_in_abstract_count=0
            print(filename)
            f = open(TEST_SQL_INPUT + filename, 'r',encoding='utf-8')
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
            #f2.write('title_score')
            #f2.write('\t')
            f2.write('title_abstract_score')
            f2.write('\t')
            f2.write('title_coverage')
            f2.write('\t')
            f2.write('abstract_coverage')
            f2.write('\t')
            f2.write('rank')
            f2.write('\n')
            rank=1
            for line in f.readlines()[1:]:
                line = line.strip()
                #print(line[:20])
                items = line.split('\t')
                sentence = text_to_wordlist(items[0])
                sentence_pmid=int(items[1])
                    
                pmid = int(items[2])
                try:
                    citation_pmid = int(items[3])
                except:
                    print("No sentence:")
                    print(line)
                    continue
                    
                ArticleTitle=text_to_wordlist(items[5])
                abstract=text_to_wordlist(items[6])
             
                #title_score=float(items[6])
                title_abstract_score=float(items[4])
                title_coverage=float(items[8])
                abstract_coverage=float(items[9])
                rank=int(items[7])
               
                if pmid in indexs:
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
                    #f2.write(str(title_score))
                    #f2.write('\t')
                    f2.write(str(title_abstract_score))
                    f2.write('\t')
                    f2.write(str(title_coverage))
                    f2.write('\t')
                    f2.write(str(abstract_coverage))
                    f2.write('\t')
                    f2.write(str(rank))
                    f2.write('\n')       
                    rank+=1

            f2.close()
            f.close()
            end=time.time()
            print("time: %f"%(end-start))
            
if __name__ == "__main__":
    BASE_DIR = 'input/'
    
    
    TEST_SQL_INPUT = '../DL_testing_final_keyword_only/'
    
    output = 'case_SQL_testing_final_complete_keywordOnly/'
    
    citations_dict=load_json_citation(BASE_DIR+'abstract_citations_by_year.json')
    
    abs_citation_IF_input_file = BASE_DIR + 'title_abstracts_complete_only_features.tsv'
    
    print("read %s:"%abs_citation_IF_input_file,flush=True)
    
    reader_abs_citation_IF=pd.read_csv(abs_citation_IF_input_file, encoding='utf-8',sep='\t', 
                                       index_col='pmid')
    print(reader_abs_citation_IF.shape)
    reader_abs_citation_IF = reader_abs_citation_IF.loc[~reader_abs_citation_IF.index.duplicated(keep='first')]
    print(reader_abs_citation_IF.shape)
    
    normalized_citation_by_year=load_normalized_citation_by_year(BASE_DIR)
    create_SQL_final_test(TEST_SQL_INPUT,output,reader_abs_citation_IF,citations_dict,normalized_citation_by_year)