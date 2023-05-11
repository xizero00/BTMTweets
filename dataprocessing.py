import json
import string
import pickle
import numpy as np
from xml.dom.minidom import Document
#　tweet-preprocessor
import preprocessor.api as p
# https://towardsdatascience.com/basic-tweet-prconda install -c saidozcan tweet-preprocessoreprocessing-in-python-efd8360d529e

from nltk import download
from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer


# sklearn
from sklearn.feature_extraction.text import CountVectorizer

stop_words = stopwords.words('english')
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def load_data(datapath, suffix):
    if suffix == 'json':
        with open(datapath) as fd:
            data = json.load(fd)
    elif suffix == 'txt':
        fd = open(datapath)
        data = fd.readlines()
    return data

def preprocessing_json(data):
    document = []
    for d in data:
        # Preprocessing: remove hashtag, hyperlinks, Mentions, Emojis and so on!
        d['pt'] = p.clean(d['full_text'])

        # Remove punctuation
        d['pt'] = "".join([char for char in d['pt'] if char not in string.punctuation])
        
        # Tokenize
        # d['pt'] = p.tokenize(d['pt']) # this is different from conventional tokenization method
        d['pt'] = word_tokenize(d['pt'])
        # Remove stop words
        d['pt'] = [word for word in d['pt'] if word not in stop_words]
        # Stemming
        d['pt'] = [porter.stem(word) for word in d['pt']]
        # Lemmatize
        d['pt'] = [lemmatizer.lemmatize(word) for word in d['pt']]
        document.append(d['pt'])

    return data, document

def preprocessing_txt(data):
    document = []
    i = 1
    total = len(data)
    for d in data:
        # Preprocessing: remove hashtag, hyperlinks, Mentions, Emojis and so on!
        d = p.clean(d)

        # Remove punctuation
        d = "".join([char for char in d if char not in string.punctuation])
        
        # Tokenize
        # d['pt'] = p.tokenize(d['pt']) # this is different from conventional tokenization method
        d = word_tokenize(d)
        # Remove stop words
        d = [word for word in d if word not in stop_words]
        # Stemming
        d = [porter.stem(word) for word in d]
        # Lemmatize
        d = [lemmatizer.lemmatize(word) for word in d]
        if len(d) == 0:
            print('wrong')
        else:
            document.append(d)
        if (i % 50 == 0):
            print(f"{i}/{total} finished")
            # break
        i += 1

    return data, document

if __name__ == "__main__":
    download('stopwords')
    filename = 'dataset_clean_english_only_new.txt'
    datapath = './data/' + filename
    suffix = filename.split('.')[1]

    data = load_data(datapath, suffix)

    if suffix == 'json':
        data, doc = preprocessing_json(data)
    elif suffix == 'txt':
        data, doc = preprocessing_txt(data)
    
    with open(datapath + 'after_preprocess_' + filename, 'w') as fd:
        for d in doc:
            res = ""
            for w in d:
                res += w + " "
            fd.write(res + '\n')