import codecs
from typing import List
import os

data_paths = ['ming1.csv', 'tang.csv', 'ming2.csv', 'ming3.csv', 'ming4.csv', 'qing1.csv', 'qing2.csv']
stop_word_path = 'stopwords.txt'
sxhy_path = os.path.join('./data/', 'sxhy_dict.txt')
VOCAB_SIZE = 6000

'''
poem["title"]
poem["author"]
poem[sentence] => len ==4
'''

def get_corpus() -> List:
    corpus = []
    for data_path in data_paths:
        data = read_corpus_file('./data/'+data_path)
        corpus.extend(data)
    print(len(corpus))
    return data

def get_stopwords() -> List:
    stopwords = set()
    with codecs.open('./data/'+stop_word_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            stopwords.add(line.strip())
            line = fin.readline()
    return stopwords

def read_corpus_file(path):
    contents = codecs.open(path, 'r', 'utf-8').read()
    result = []
    for poem in contents.split('\n'):
        if not poem:
            continue
        new_item = {'sentence':[]}
        sentences = poem.split(',')[3]
        sentences = sentences[1:-3].split('。')
        if len(sentences)!=2:
            continue

        for sentence in sentences:
            half = sentence.split('，')
            if len(half)!=2:
                continue
            new_item['sentence'].extend(half)
        if len(new_item['sentence'])==4:
            result.append(new_item)

    return result

def get_sxhy_dict():
    sxhy_dict = set()
    with codecs.open(sxhy_path, 'r', 'utf-8') as fin:
        line = fin.readline()
        while line:
            sxhy_dict.add(line.strip())
            line = fin.readline()
    return sxhy_dict
if __name__ == '__main__':
    get_corpus()