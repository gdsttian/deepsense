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

Full-text articles downloaded from PMC were split into three sets of articles for developing training, validation and test datasets. An SQL database was built to store all the downloaded PubMed abstracts which were indexed for query by a modified BM25 algorithm.

Sentences with citations were extracted from the full-text part of the PMC XML files and were paired with the cited articles with titles, abstracts, publication year, article type and journal names.

To create the training and validation datasets, for each pair of citing sentence and cited article extracted from PMC full-text articles for training and validation, the citing sentence was used to query the SQL database to obtain top 1,000 query results. Two negative cases were selected for each sentence query: one case was randomly chosen from the top 1,000 query results excluding the cited article, and the other case was randomly chosen from all the abstracts outside of the top 1,000 results. Each citing sentence was then paired with the cited articles and the negative cases for creating the training and validation datasets. After that, use the script `create_SQL_final_train_valid.py` in the folder `scripts` to generate the final training and validation datasets.

Several test datasets were developed for model evaluation. For each pair of citing sentence and cited article extracted from PMC full-text articles for test, the citing sentence was used to query the SQL database, PubMed TF-IDF API and PubMed BestMaatch API to get the top 1,000 returns for developing the D1, D2, D3 test datasets respectively. For development of D1, we kept the sentences whose cited articles are included in the top 1,000 search results, and used the script `create_SQL_final_test.py` in folder `script` to generate D1. For development of D2, the script `create_test_search_PubMed.py` in folder `script` was used to get the top 1,000 query results for searching the citing sentences through the PubMed TF-IDF API. The script `create_PM_final_test.py` in folder `script` was then used to generate D2. The notebooks `search_engine_project_200825.ipynb` and `deepsense_biomedical_literature_sentence_search.ipynb` in folder `script` include code snippet for gnerating D3 and final processing of D1, D2 and D3.

Sentences for query of Google Scholar were randomy sampled from D3. Those sentences were included in file `sentences_for_google_scholar_search.csv` in folder `data`.


### TREC-COVID Dataset

The TREC-COVID dataset was processed using code snippets in the notebook `trec_covid_data_process.ipynb` in folder `scripts`.


## Word Embedding Training, DeepSenSe Model Training and Evaluation

### Word Embedding

Word embeddings were developed using the `fastText` package, which can be downloaded at https://fasttext.cc/.


### DeepSenSe Model Training

Use the script `train_sentences_new.py` in folder `model`, and run `python train_sentences_new.py` to train the DeepSenSe model.


### Evaluation

Use the script `test_sentences_sql_bm25.py` in folder `model`, and run `python test_sentences_sql_bm25.py` to get re-ranking results of D1.

Use the script `test_sentences_pubmed_tfidf.py` in folder `model`, and run `python test_sentences_pubmed_tfidf.py` to get re-ranking results of D2.

Use the script `test_sentences_pubmed_bm.py` in folder `model`, and run `python test_sentences_pubmed_bm.py` to get re-ranking results of D3.

Use the script `test_sentences_trec_covid_bm25.py` in folder `model`, and run `python test_sentences_trec_covid_bm25.py` to get re-ranking results of TREC-COVID dataset.

Use the notebook `deepsense_performance_calculation.ipynb` in folder `scripts` for calculation of evaluation metrics for D1, D2, D3.

Evaluation metrics on the TREC-COVID dataset were computed using the [trec_eval](https://github.com/usnistgov/trec_eval) scoring program.

