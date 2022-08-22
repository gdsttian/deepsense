'''
sentence	sentence_pmid	result_pmids	citation_pmid	title_abstract	title_abstract_score	title_score	title	abstract	sql_rank	title_coverage	abstract_coverage	sentence_orig
'''


import os
import inflection

path = '../case_DL_testing_final_with_score_coverage/'
output = '../case_DL_testing_final_sentence_with_all_keywords_score_coverage/'

if not os.path.exists(output):
    os.mkdir(output)


def find_keyword_sentence(sentence, title_abstract):
    terms_in_abstract = [inflection.singularize(i) for i in set(title_abstract.split())]
    keyword_sentence = [word for word in sentence.split() if inflection.singularize(word) in terms_in_abstract]
    return ' '.join(keyword_sentence)

for filename in sorted(os.listdir(path)):
    sentence_keyword_sentece_dic = {}
    print(filename)
    f2 = open(output + filename, 'w')

    f = open(path + filename, 'r')
    f2.write(f.readlines()[0])
    f.close()

    f = open(path + filename, 'r')
    for line in f.readlines()[1:]:
        terms = line.strip().split('\t')
        sentence = terms[0]
        result_pmid = terms[2]
        citation_pmid = terms[3]
        title_abstract = terms[4]
        if result_pmid == citation_pmid:
            keyword_sentence = find_keyword_sentence(sentence, title_abstract)
            sentence_keyword_sentece_dic[sentence] = keyword_sentence
    f.close()

    f = open(path + filename, 'r')
    for line in f.readlines()[1:]:
        terms = line.split('\t')
        sentence = terms[0]
        if sentence in sentence_keyword_sentece_dic.keys():
            terms[0] = sentence_keyword_sentece_dic[sentence]
            f2.write('\t'.join(terms))
    f.close()

    f2.close()
