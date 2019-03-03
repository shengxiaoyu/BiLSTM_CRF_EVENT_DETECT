__doc__ = 'description'
__author__ = '13314409603@163.com'


BDFH = {'、', ',', '.', '，', '。', '!', '！', '?', '？', ';', '；', ':', '：', '-', '—', '(', '（', ')', '）', '《', '》',
                '“', '”'}
class DataProcess(object):
    # 标点符号集
    def getContent(self):
        raise NotImplementedError()

    def segmentAndSave(self):
        raise NotImplementedError()

    def delBdfh(self,words):
        newWords = []
        for word in words:
            if (word not in BDFH):
                newWords.append(word)
        return newWords


if __name__ == '__main__':
    pass