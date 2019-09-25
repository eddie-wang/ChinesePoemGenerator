
import jieba
import os
from utils import  get_sxhy_dict

sxhy_path = os.path.join('./data/', 'sxhy_dict.txt')

look_forward_len=3


class Segmenter:
    def __init__(self) -> None:

        jieba.load_userdict(sxhy_path)
        self._sxhy_dict = get_sxhy_dict()

    def segment(self, sentence):
        segs = set()
        for i in range(len(sentence)):
            for k in range(1,look_forward_len):
                if i+k<=len(sentence) and sentence[i:i+k] in self._sxhy_dict:
                    segs.add(sentence[i:i+k])

        for i in range(0, len(sentence), 2):
            if i+3 >= len(sentence):
                seg_list = jieba.lcut(sentence[i:], HMM=True, cut_all=True)
            else:
                seg_list = jieba.lcut(sentence[i:i+2], HMM=True, cut_all=True)
            segs.update(seg_list)
        return list(segs)



if __name__ == '__main__':
    segmenter = Segmenter()
    print(segmenter.segment(u'鹧鸪声里雨班班'))