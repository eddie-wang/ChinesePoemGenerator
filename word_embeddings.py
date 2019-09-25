from gensim.models import Word2Vec
from utils import get_corpus, VOCAB_SIZE
import numpy as np

class WordEmbedder:

    def __init__(self) -> None:
        super().__init__()
        self.corpus = get_corpus()
        self.int2ch, self.ch2int = self.get_vabulary(self.corpus)
        self.embeddings = {}

    def get_word_embedding(self, ndim):
        return self.generate_word2vec_embedds(ndim)

    def generate_word2vec_embedds(self, ndim):
        if len(self.embeddings)>0:
            return self.embeddings
        sentences = []
        for poem in self.corpus:
            last_chars = []
            for sentence in poem['sentence']:
                sentences.append(list(filter(lambda ch: ch in self.ch2int, sentence)))
                last_chars.append(sentence[-1])
            sentences.append(last_chars)
        model = Word2Vec(sentences, size = ndim, min_count = 5)
        # print(model.similar_by_word('水'))
        embeddings = np.random.uniform(-1,1,[VOCAB_SIZE, ndim])

        for idx, ch in enumerate(self.int2ch):
            if ch in model.wv:
                embeddings[idx, : ] = model.wv[ch]
        self.embeddings = embeddings
        return embeddings

    def get_vabulary(self, corpus):
        ch_count={}
        for poem in self.corpus:
            for sentence in poem['sentence']:
                for ch in sentence:
                    ch_count[ch] = ch_count[ch]+1 if ch in ch_count else 1

        vocab = sorted([ch for ch in ch_count], key=lambda ch: -ch_count[ch])[:VOCAB_SIZE - 2]
        vocab.insert(0,u'<START>')
        vocab.append(u'<PAD>')
        return vocab, dict((ch, idx) for idx, ch in enumerate(vocab))


if __name__ == '__main__':
    word_embedder = WordEmbedder()
    embedds = word_embedder.get_word_embedding(512)