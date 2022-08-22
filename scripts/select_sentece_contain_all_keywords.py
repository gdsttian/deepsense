'''
sentence	sentence_pmid	citation_pmid	title	abstract	label	title_score	title_abstract_score	title_coverage
	abstract_coverage
'''

import os

path = '../case_with_all_label_score_true_in_top10000_add_true_case_with_keyword_only2/'
output = 'case_sentence_contain_all_keyword_only/'

if not os.path.exists(output):
    os.mkdir(output)

for filename in sorted(os.listdir(path)):
    print(filename)
    f2 = open(output + filename, 'w')
    f = open(path + filename, 'r')
    f2.write(f.readlines()[0])
    f.close()
    f = open(path + filename, 'r')
    pre_line = ''
    cur_line = ''
    for line in f.readlines()[1:]:
        cur_line = line
        cur_terms = cur_line.strip().split('\t')
        cur_sentence = cur_terms[0]
        cur_sentence_pmid = cur_terms[1]
        cur_citation_pmid = cur_terms[2]
        pre_sentence = ''
        pre_sentence_pmid = ''
        pre_citation_pmid = ''
        if len(pre_line) > 0:
            pre_terms = pre_line.strip().split('\t')
            pre_sentence = pre_terms[0]   
            pre_sentence_pmid = pre_terms[1]
            pre_citation_pmid = pre_terms[2]
        if cur_sentence_pmid == pre_sentence_pmid and cur_citation_pmid == pre_citation_pmid and len(cur_sentence) <= len(pre_sentence):
            f2.write(cur_line)
        pre_line = cur_line
    f.close()
    f2.close()
