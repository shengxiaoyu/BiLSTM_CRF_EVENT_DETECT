import os

__doc__ = 'description:起诉状处理'
__author__ = '13314409603@163.com'

from data_pre_process.data_process import DataProcess
from enum import Enum
from pyltp import SentenceSplitter
import re

class QszProcess(DataProcess):
    def __init__(self,rootDir,):
        self.rootDir = rootDir
    def getContent(self):
        raise NotImplementedError()

    def segmentAndSave(self,segmentor,segment_result_save_path):
        for root, dirs, files in os.walk(self.rootDir):
            for dir in dirs:
                content = self.__get_content_of_one_Qsz__(os.path.join(self.rootDir, dir))
                ssly = self.__getSsly__(content)
                # 排除空
                if(ssly==None):
                    print(dir+' '+content)
                else:
                    # 分句
                    sentences = SentenceSplitter.split(ssly)
                    lines = []
                    for str in sentences:
                        # 分词
                        words = segmentor.segment(str)
                        # 去除标点符号
                        words = self.delBdfh(words)
                        if (len(words) == 0):
                            continue
                        lines.append(words)
                    if(len(lines)>0):
                        fw = open(os.path.join(segment_result_save_path, dir + '.txt'), 'w', encoding='utf8')
                        for words in lines:
                            fw.write(' '.join(list(words)))
                            fw.write('\n')
                        fw.flush()
                        fw.close()
        segmentor.release()
        return

    # 获得单个起诉状内容
    def __get_content_of_one_Qsz__(self,path):
        # 单个案件地址
        #获取该案件地址下的file
        files = [fs for rt, ds, fs in os.walk(path)][0]
        content = ''
        for fileName in files:
            if (fileName.find('_ocr.txt') != -1):
                with open(os.path.join(path,fileName),encoding='utf8') as f:
                    content += f.read()
        #去掉换行符，制表符，空格，这些可能是ocr识别错误生成的
        content = content.replace("\n", '').replace("\r\n", '').replace('\t', '').replace(' ', '')
        return content

    # 获取事实和理由段
    def __getSsly__(self,content):
        beginIndex = -1;
        for item in sslyBeginEnum:
            if (content.find(item.value) != -1):
                beginIndex = content.find(item.value) + len(item.value) + 1
                break
        if (beginIndex == -1):
            return

        endIntex = -1;
        for item in sslyEndEnum:
            if (content.find(item.value) != -1):
                endIntex = content.find(item.value)
                break
        if (endIntex == -1):
            for item in sslyEndEnum2:
                if (content.find(item.value) != -1):
                    endIntex = content.find(item.value) + len(item.value)
                    break
        if (endIntex == -1):
            return
        ssly = content[beginIndex:endIntex]
        return ssly

    # 获取诉讼请求
    def __getSsqq__(self,content):

        # 获取起止index
        benginIndex = 0
        endIndex = 0
        for index, str in enumerate(content):
            if (str.find('诉讼请求') != -1):
                benginIndex = index
            if (str.find('事实和理由') != -1):
                endIndex = index

        # 获取诉讼请求内容
        ssqqList = []
        for str in content[benginIndex + 1:endIndex]:
            re_begin = re.compile(r'^[1-9一二三四五六七八九]+、')
            if (re_begin.match(str)):
                ssqqList.append(str)
            else:
                ssqqList.append(ssqqList.pop() + str)
        return ssqqList

#事实与理由开头的关键词
class sslyBeginEnum(Enum):

    v1 = '事实与理'
    v2 = '事实和理'
    v3 = '事实理'
    v4 = '事实及理'
    v5 = '事实！理由'
    v6 = '事头理由'
    v7 = '事火及理由'
    v8 = '事实、理由'
    v9 = '事与理山'
    v10 = '实及理由'
    v11 = '事头及理'
    v12 = '请求事项'
    v13 = '事实！;理'
    v14 = '事实\'j理'
    v15 = '事山与理'
    v16 = '事实Jj理'
    v17 = '实施与理'
    v18 = '事实1j玲'
    v19 = '事火与理'
    v20 = '事实及现出'
    v21 = '小头及理由'
    v22 = '事实及法律根据'
    v23 = '实事与理由'
    v24 = '小实和理由'
    v25 = '瓜火和理||'
    v26 = '事实及原'
    v27 = '头与理由'
    v28 = '事头和理'
    v29 = '实施及理'

    # v1 = '事实与理由'
    # v2 = '事实和理由'
    # v3 = '事实理由'
    # v4 = '事实及理由'

    # v5 = '事实与理出'
    # v6 = '事实和理出'
    # v7 = '事实理出'
    # v8 = '事实及理出'
    #
    # v9 = '事实与理山'
    # v10 = '事实和理山'
    # v11 = '事实理山'
    # v12 = '事实及理山'
    #
    # v13 = '事实与理巾'
    # v14 = '事实和理巾'
    # v15 = '事实理巾'
    # v16 = '事实及理巾'

#事实与理由结尾关键词1
class sslyEndEnum(Enum):
    v1 = '此致'
    v2 = '此  致'
    v4 = '此呈'
    v5 = '此  呈'
    v6 = '此，致'
    v7 = '匕致'
    v8 = '比致'

#事实与理由结尾关键词2
class sslyEndEnum2(Enum):
    v1='望依法判决'
    v2 = '判如所请'
    v3 = '判如原告所请'
    v4 = '判如诉请'

    v5 = '判决双方离婚'
    v6 = '判决准予离婚'
    v7 = '判决原告被告解除婚姻关系'
    v8 = '请求法院尽快判决离婚'
    v9 = '判准原告与被告离婚'
    v10 = '判令原被告离婚'
    v11 = '依法准予原被告离婚'


    v12 = '特提起离婚诉讼'
    v13 = '请求法院判准'
    v14 = '请求法院准予'

    v15 = '支持原告全部的诉讼请求'
    v16 = '支持原告的诉讼请求'
    v17 = '支持原告上述诉讼请求'
    v18 = '依法支持原'
    v19 = '支持原告上述诉请'
    v20 = '依法支持诉请'
    v21 = '依法判准原告以上请求'
    v22 = '支持原告诉请'
    v23 = '支持原告的请求'
    v24 = '支持原告（我）的诉讼请求'
    v25 = '支持原告离婚诉求'

    v26 = '请求人民法院批准为盼'

    v27 = '故请求人民法院判准原、被告离婚'
    v28 = '无法再继续生活下去，因此原告起诉与被告离婚'
    v29 = '特向人民法院提起诉讼'
    v30 = '故再次诉至贵院依法裁决数悉'
    v31 = '请求法院判决'
    v32 = '诉讼费由被告人陌担，请法院判予离婚诉求'
    v33 = '依法判准原告的诉讼请求'
    v34 = '望贵院作出公正判决'
    v35 = '恳请法院判准原告的离婚之情'



if __name__ == '__main__':
    pass