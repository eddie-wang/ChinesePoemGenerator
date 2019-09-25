import codecs
import json
import os
from segmenter import Segmenter
from utils import get_corpus, get_stopwords, get_sxhy_dict
from functools import cmp_to_key

rank_path = 'rank.json'

def _text_rank(adjlist):
    damp = 0.85
    scores = dict((word, 1.0) for word in adjlist)
    try:
        for i in range(100000):

            cnt = 0
            new_scores = dict()
            for word in adjlist:
                new_scores[word] = (1 - damp) + damp * sum(adjlist[other][word] * scores[other] \
                                                           for other in adjlist[word])
                if scores[word] == new_scores[word]:
                    cnt += 1

            print("Done (%d/%d)" % (cnt, len(scores)))
            if len(scores) == cnt:
                break
            else:
                scores = new_scores

    except KeyboardInterrupt:
        pass
    sxhy_dict = get_sxhy_dict()

    def _compare_words(a, b):
        if a[0] in sxhy_dict and b[0] not in sxhy_dict:
            return -1
        elif a[0] not in sxhy_dict and b[0] in sxhy_dict:
            return 1
        else:
            return b[1]-a[1]

    words = sorted([(word, score) for word, score in scores.items()],
                   key=cmp_to_key(_compare_words))
    with codecs.open('./data/'+rank_path, 'w', 'utf-8') as fout:
        json.dump(words, fout)


def get_text_ranks():
    segmenter = Segmenter()
    stopwords = get_stopwords()
    print("Start TextRank over the selected quatrains ...")
    corpus = get_corpus()
    adjlist = dict()
    for idx, poem in enumerate(corpus):
        if 0 == (idx + 1) % 10000:
            print("[TextRank] Scanning %d/%d poems ..." % (idx + 1, len(corpus)))
        for sentence in poem['sentence']:
            segs = list(filter(lambda word: word not in stopwords,
                          segmenter.segment(sentence)))
            for seg in segs:
                if seg not in adjlist:
                    adjlist[seg] = dict()

            for i, seg in enumerate(segs):
                for _, other in enumerate(segs[i + 1:]):
                    if seg != other:
                        adjlist[seg][other] = adjlist[seg][other] + 1 \
                            if other in adjlist[seg] else 1.0
                        adjlist[other][seg] = adjlist[other][seg] + 1 \
                            if seg in adjlist[other] else 1.0

    for word in adjlist:
        w_sum = sum(weight for other, weight in adjlist[word].items())
        for other in adjlist[word]:
            adjlist[word][other] /= w_sum
    print("[TextRank] Weighted graph has been built.")
    _text_rank(adjlist)

def get_word_ranks():
    if not os.path.exists('./data/rank.json'):
        get_text_ranks()
    with codecs.open('./data/' + rank_path, 'r', 'utf-8') as fin:
         ranks = json.load(fin)
    return dict((pair[0], pair[1]) for idx, pair in enumerate(ranks))

if __name__ == '__main__':
    print(get_word_ranks())