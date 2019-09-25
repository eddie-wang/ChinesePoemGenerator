import codecs
from functools import reduce

from segmenter import Segmenter
from text_rank import get_word_ranks
from utils import  get_corpus

def prepare_training_data():
    word_ranks = get_word_ranks()
    corpus = get_corpus()
    segmenter = Segmenter()
    with codecs.open('./data/training.txt', 'w', 'utf-8') as fout:
        for poem in corpus:
            poem['keyword'] = []
            stop=False

            for sentence in poem['sentence']:
                segs = list(filter(lambda seg: seg in word_ranks, segmenter.segment(sentence)))
                if len(segs) == 0:
                    stop = True
                    break
            if len(poem['sentence'])!=4 or stop:
                continue
            for sentence in poem['sentence']:
                segs = list(filter(lambda seg: seg in word_ranks, segmenter.segment(sentence)))
                if len(segs) == 0:
                    print('aaa', sentence)
                keyword = reduce(lambda x,y: x if word_ranks[x]>word_ranks[y] else y, segs)
                poem['keyword'].append(keyword)
                if(len(keyword)>=2):
                    print(sentence, keyword)
                fout.write(sentence + '\t' + keyword + '\n')

if __name__ == '__main__':
    prepare_training_data()
    # word_ranks = get_word_ranks()
    # for word in word_ranks:
    #     if(len(word)>3):
    #         print(word)