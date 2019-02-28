__doc__ = 'description:封装word2vec模型训练'
__author__ = '13314409603@163.com'

import os
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

class Word2VecModel(object):
    def __init__(self,model_save_path,train_data_save_path,size,window=5,min_count=2,workers=4):
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
            self.model = Word2Vec(PathLineSentences(train_data_save_path),size=self.size,window=self.window,min_count=self.min_count,workers=self.workers)
            print('word2vec model first train end!')

            self.save()
            print('word2vec model save :'+self.save_path)

    def reTrain(self,train_data_save_path):
        # self.model.build_vocab(MySentences(train_data_save_path)
        self.model.train(PathLineSentences(train_data_save_path))
        print('word2vec retrain end!')

    def save(self):
        self.model.save(self.save_path)

    def getEmbedded(self):
        return self.model.wv

# class MySentences(object):
#     def __init__(self,dir):
#         self.dir = dir
#     def __iter__(self):
#         for fname in os.listdir(self.dir):
#             for line in open(os.path.join(self.dir,fname),encoding='utf8'):
#                 yield line.strip()

if __name__ == '__main__':
    pass