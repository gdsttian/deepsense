# -*- coding: utf-8 -*-
"""
API function for PubMed
"""

from Bio import Entrez
import pandas as pd
import time
from interruptingcow import timeout

Entrez.email = "cl17d@my.fsu.edu"

retmax=1000


def create_data_PubMed(sen_pmids,TEST_OUT_FILE,LOG_FILE,op=None):
    
    texts = [] 
    i=0

    for sen,sen_pmid,cite_pmid,year,sentence_orig in sen_pmids:

        i=i+1

        if op.upper()=='OR':
            text = sen.split()
            text = " OR ".join(text)
        else:
            text=sen
        try:
            with timeout(60, exception = RuntimeError): #only for linux

                handle = Entrez.esearch(db="pubmed", 
                        sort='relevance', 
                        retmax=retmax,
                        retmode='xml', 
                        usehistory="y",
                        api_key="a1b01216ee1c75911b845857357734e25007",
                        datetype='pdat',
                        mindate=1977,
                        maxdate=year,
                        term=text)
                results = Entrez.read(handle)
                handle.close()
                id_list = results['IdList'] # pmids of search result
                dup=False
                k=1
                time.sleep(0.1)
                
                for j in id_list:

                    texts.append([sen,str(sen_pmid),str(j),str(cite_pmid),sentence_orig])
                    k=k+1
                    if int(j)==cite_pmid:
                        
                        dup=True
                    if k >retmax:
                        break
                if dup ==False:
                    
                    texts.append([sen,str(sen_pmid),'99999',str(cite_pmid),sentence_orig]) #cited article is not in search result
                
        except RuntimeError:

            with open(LOG_FILE,'a') as f:
                f.write(str(i)+' timeout\n')
            
        except:
            try:
                with open(LOG_FILE,'a',encoding='utf-8') as f:
                  
                    f.write(str(i)+' '+ sen+' '+str(cite_pmid)+' fail\n')
            except:
                with open(LOG_FILE,'a',encoding='utf-8') as f:
                    f.write(str(i)+' fail\n')
        if i%100==0:
            with open(LOG_FILE,'a') as f:
                f.write(str(i)+'\n')
    with open(TEST_OUT_FILE,'w',encoding='utf-8_sig') as f:
        
        for text in texts:
            f.write('\t'.join(text)+'\n')


