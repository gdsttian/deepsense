import pandas as pd
import os
import re
import gc
import time
import random 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

greek_alphabet = {
    u'\u0391': 'Alpha',
    u'\u0392': 'Beta',
    u'\u0393': 'Gamma',
    u'\u0394': 'Delta',
    u'\u0395': 'Epsilon',
    u'\u0396': 'Zeta',
    u'\u0397': 'Eta',
    u'\u0398': 'Theta',
    u'\u0399': 'Iota',
    u'\u039A': 'Kappa',
    u'\u039B': 'Lamda',
    u'\u039C': 'Mu',
    u'\u039D': 'Nu',
    u'\u039E': 'Xi',
    u'\u039F': 'Omicron',
    u'\u03A0': 'Pi',
    u'\u03A1': 'Rho',
    u'\u03A3': 'Sigma',
    u'\u03A4': 'Tau',
    u'\u03A5': 'Upsilon',
    u'\u03A6': 'Phi',
    u'\u03A7': 'Chi',
    u'\u03A8': 'Psi',
    u'\u03A9': 'Omega',
    u'\u03B1': 'alpha',
    u'\u03B2': 'beta',
    u'\u03B3': 'gamma',
    u'\u03B4': 'delta',
    u'\u03B5': 'epsilon',
    u'\u03B6': 'zeta',
    u'\u03B7': 'eta',
    u'\u03B8': 'theta',
    u'\u03B9': 'iota',
    u'\u03BA': 'kappa',
    u'\u03BB': 'lamda',
    u'\u03BC': 'mu',
    u'\u03BD': 'nu',
    u'\u03BE': 'xi',
    u'\u03BF': 'omicron',
    u'\u03C0': 'pi',
    u'\u03C1': 'rho',
    u'\u03C3': 'sigma',
    u'\u03C4': 'tau',
    u'\u03C5': 'upsilon',
    u'\u03C6': 'phi',
    u'\u03C7': 'chi',
    u'\u03C8': 'psi',
    u'\u03C9': 'omega',
}

def greek_to_eng(c):
    d=greek_alphabet.get(c)
    if d!=None:
        c=d
    return c
def clean_ref_text(text):#, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
#text='The amygdala and hippocampus are subcortical brain regions associated with ASD (Aylward et al, XREF_B22_END ; Schumann et al, XREF_B260_END ; Schumann and Amaral, XREF_B259_END )'   
    text = re.sub(r"\(.{,20}XREF_(\S){1,40}_END.{1,50}?\)",'',text)
    text = re.sub(r"XREF_(\S){1,40}_END",'',text)
    text = re.sub(r"FIG_END",'',text)
    text = re.sub(r"TABLE_END",'',text)
    text = re.sub(r"\s{2,}", " ", text)
    return(text)
def text_to_wordlist(text,remove_stopwords=True, stem_words=False):#, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
#text='The amygdala and hippocampus are subcortical brain regions associated with ASD (Aylward et al, XREF_B22_END ; Schumann et al, XREF_B260_END ; Schumann and Amaral, XREF_B259_END )'   
    text = re.sub(r"\(.{,20}XREF_(\S){1,40}_END.{1,50}?\)",'',text)
    text = re.sub(r"XREF_(\S){1,40}_END",'',text)
    text = re.sub(r"FIG_END",'',text)
    text = re.sub(r"TABLE_END",'',text)
    # Convert words to lower case and split them
    text = text.lower().split()

#    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text

    text = re.sub(r"[][#$%&()*;<>?@_`{|}~]", " ", text)

    text = re.sub(r"\.", " ", text)
    text = re.sub(r"\, ", " ", text)
    text = re.sub(r"!", " ", text)

    text = re.sub(r"\—", "-", text)
    text = re.sub(r"–", "-", text)

    text = re.sub(r"'", " ", text)
    text = re.sub(r"‘", " ", text)
    text = re.sub(r"’", " ", text)
    text = re.sub(r"“", " ", text)
    text = re.sub(r"”", " ", text)
    text = re.sub(r'\W', " ", text)

    text = re.sub(r",", " ", text)
    text = re.sub(r" et al", " ", text)
    text = re.sub(r" etc", " ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r" uk ", " united kingdom ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"\s{2,}", " ", text) # remove extra spaces
    text = text.split()
    text = " ".join(text)
    text = [greek_to_eng(w) for w in text]
    text = "".join(text)
#    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

#fuzz.ratio("Nonlinear stability analyses of vegetative pattern formation in an arid environment", "Nonlinear stability analyses of vegetative pattern formation in an arid environment")
def text_to_wordlist_case(text): 
    text = re.sub(r"\(.{,20}XREF_(\S){1,40}_END.{1,50}?\)",'',text)
    text = re.sub(r"XREF_(\S){1,40}_END",'',text)
    text = re.sub(r"FIG_END",'',text)
    text = re.sub(r"TABLE_END",'',text)
    text = re.sub(r'\W', " ", text)
    text = re.sub(r"\s{2,}", " ", text) # remove extra spaces
    text = re.sub(r" et al", " ", text)
    text = re.sub(r" etc", " ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r" uk ", " united kingdom ", text)
    text = re.sub(r"\0s", "0", text)
    text = [greek_to_eng(w) for w in text]
    text = "".join(text)
    return(text)
    
def get_closest_match(x, list_strings):

  best_match = None
  highest_jw = 90

  for current_string in list_strings:
    current_score = fuzz.ratio(x, current_string)

    if(current_score > highest_jw):
      #highest_jw = current_score
      best_match = current_string

      return best_match
  
