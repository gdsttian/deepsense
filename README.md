# DeepSenSe: A Deep Learning Model for Searching Relevant Articles using Sentences

This repo contains the codes for our paper **DeepSenSe: A Deep Learning Model for Searching Relevant Articles using Sentences**.

The codes in this repo include scripts and notebooks for data processing and computation of evaluation metrics under the directory `scripts` and scripts for traning and evaluation of the DeepSenSe under the directory `model`.

We used Python 3.7 in this study. Python packages and corresponding versions are listed in the file `requirement.txt`.

## Data Sources

The original data used for our study are available from the following sources:

- PMC Full-text Articles: https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/
- PubMed Abstracts: https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
- Journal Citation Reports: https://clarivate.com/webofsciencegroup/solutions/journal-citation-reports/
- TREC-COVID Dataset: https://ir.nist.gov/covidSubmit/

Data in the listed sources are updated from time to time. We downloaded the PMC full-text articles and PubMed abstracts on Oct. 23rd, 2019. We used the journal citation report of 2018 and the [TREC-COVID Complete](https://ir.nist.gov/covidSubmit/data.html) dataset.  


## Data Processing

Data needed for this study include a corpus for developing word embeddings, a dictionary of journal impact factors and citations of PubMed articles, a training and a validation datasets development of the DeepSenSe model, and several test datasets for evaluation of the DeepSenSe model.


### Corpus for Word Embeddings

The corpus for developing word embeddings was created from all PubMed abstracts.


### Journal Impact Factor and Citation Dictionary

The journal impact factor and citation dictionary was developed by extracting the impact factor of each journal from the journal citation report 2018 and counting citations of each article in PubMed.


### Training, Validation and Test Data

We extracted sentences with citations from the full-text part of the PMC XML files,
and obtained titles, abstracts, publication year, article type (journal
article, review, case reports, etc.) and journal names from the
MEDLINE citations to build an internal database. This internal
database is necessary since we will need to query it many times to
generate the training data.




### TREC-COVID Dataset




## Model Training and Evaluation

### Word Embedding


### DeepSenSe Model Training


### Evaluation
   
- Test Data


- TREC-COVID Dataset