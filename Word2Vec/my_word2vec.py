__doc__ = 'description:封装word2vec模型训练'
__author__ = '13314409603@163.com'

import os
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
import numpy as np

class Word2VecModel(object):
    def __init__(self,model_save_path,train_data_save_path,size,window=5,min_count=1,workers=4):
        self.save_path = os.path.join(model_save_path,'word2vec.model')
        self.size = size
        self.min_count = min_count
        self.window = window
        self.workers = workers
        if(os.path.exists(self.save_path)):
            self.model = Word2Vec.load(self.save_path)
        else:
            print('word2vec model first train')
            # sentencs =
            self.model = Word2Vec(PathLineSentences(train_data_save_path),size=self.size,window=self.window,min_count=self.min_count,workers=self.workers,
                                  sg=1)#1-skip,0-cbow
            print('word2vec model first train end!')

            self.save()
            print('word2vec model save :'+self.save_path)

    def reTrain(self,train_data_save_path):
        # self.model.build_vocab(MySentences(train_data_save_path)
        total_examples = 0
        for file in os.listdir(train_data_save_path):
            with open(os.path.join(train_data_save_path,file),'r',encoding='utf8')as f:
                total_examples += len(f.readlines())
        self.model.train(PathLineSentences(train_data_save_path),total_examples=total_examples,epochs=5)
        print('word2vec retrain end!')

    def save(self):
        self.model.save(self.save_path)

    def getEmbedded(self):
        return self.model.wv


if __name__ == '__main__':
    rootdir = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\'
    dim = 30
    word2vec_model_save_path = os.path.join(rootdir,'word2vec')
    wv = Word2VecModel(word2vec_model_save_path, '', 30)
    wv = wv.getEmbedded()
    wv.add('<pad>',np.zeros((dim)))
    print(wv.most_similar('原告'))
    print(wv.similarity('原告', '被告'))
    print(wv['原告'])
    print(wv['被告'])
    pass