import os
import sys

__doc__ = 'description'
__author__ = '13314409603@163.com'
from Config.config_parser import getParser
import event_dectect.First_For_Commo_Tags.config_center as CONFIG
import event_dectect.Second_For_Fine_Tags.config_center as NEW_CONFIG
from event_dectect.Event_Model.extract_event import Extractor
from event_dectect.Event_Model.model import Event
from event_dectect.Event_Model.model import parties

def run():
    FLAGS = getParser()
    CONFIG.init(FLAGS.root_dir)
    NEW_CONFIG.initNewTags(os.path.join(FLAGS.root_dir, 'full_trigger_labels.txt'),
                           os.path.join(FLAGS.root_dir, 'full_argu_labels.txt'))
    base_dir = FLAGS.labeled_data_path+'_for_second'

    true_events = []
    predict_examples = []

    test_index = FLAGS.test_folder
    test_index = test_index.split(',')
    for sub_dir_name in os.listdir(base_dir):
        sub_dir = os.path.join(base_dir,sub_dir_name)
        for index in os.listdir(sub_dir):
            if(index not in test_index):
                continue
            target_dir = os.path.join(sub_dir,index)
            for fileName in os.listdir(target_dir):
                with open(os.path.join(target_dir, fileName), 'r', encoding='utf8') as f:
                    '''把相同句子的事件抽取来放在一个数组，'''
                    # 当前句子包含的所有事件
                    events = []

                    #为了调用model.form_events方法，需要构造如下的数组相同句子的句子、第二层标记
                    words_list = []
                    second_tags_list = []
                    sentences = []
                    index_pairs_list = []
                    speakers = []


                    # 当前句子
                    currentSentence = f.readline()
                    currentWords = currentSentence.strip().split()
                    words_list.append(currentWords)

                    sentences.append(currentSentence)

                    index_pairs = []
                    base_index=0
                    for word in currentWords:
                        index_pairs.append([base_index,base_index+len(word)])
                        base_index+=len(word)
                        base_index +=1
                    index_pairs_list.append(index_pairs)

                    speakers.append(parties.PLAINTIFF)

                    # 第一层标签真实情况
                    current_first_tags = f.readline().strip().split()
                    # pos方式
                    current_pos_tags = f.readline().strip().split()

                    #第二层标记情况
                    second_tags_list.append(f.readline().strip().split())

                    # 下一个句子
                    sentence = f.readline()
                    while (sentence):
                        if (sentence == currentSentence):
                            '''同一个句子'''
                            # 加入待抽取队列
                            words_list.append(sentence.strip().split())

                            sentences.append(sentence)

                            index_pairs = []
                            base_index = 0
                            for word in currentWords:
                                index_pairs.append([base_index, base_index + len(word)])
                                base_index += len(word)
                                base_index+= 1
                            index_pairs_list.append(index_pairs)

                            speakers.append(parties.PLAINTIFF)

                            # 去掉第一层标签行和pos行
                            f.readline()
                            f.readline()

                            second_tags_list.append(f.readline().strip().split())
                        else:
                            '''不是同一个句子'''
                            #抽取所有事件
                            events = Event.form_events(words_list, second_tags_list, sentences, index_pairs_list, speakers)
                            # 将事件加入事件集
                            true_events.append(events)

                            # 合并tag，构造模型抽取的训练集
                            predict_examples.append([currentWords, current_first_tags, current_pos_tags,sentences[0],index_pairs_list[0],

                                                     [0]])

                            # 初始化
                            # 当前句子包含的所有事件
                            events = []

                            # 为了调用model.form_events方法，需要构造如下的数组相同句子的句子、第二层标记
                            words_list = []
                            second_tags_list = []
                            sentences = []
                            index_pairs_list = []
                            speakers = []

                            # 当前句子
                            currentSentence = sentence
                            currentWords = currentSentence.strip().split()
                            words_list.append(currentWords)

                            sentences.append(currentSentence)

                            index_pairs = []
                            base_index = 0
                            for word in currentWords:
                                index_pairs.append([base_index, base_index + len(word)])
                                base_index += len(word)
                                base_index+=1
                            index_pairs_list.append(index_pairs)

                            speakers.append(parties.PLAINTIFF)

                            # 第一层标签真实情况
                            current_first_tags = f.readline().strip().split()
                            # pos方式
                            current_pos_tags = f.readline().strip().split()

                            # 第二层标记情况
                            second_tags_list.append(f.readline().strip().split())


                        # 下一个句子
                        sentence = f.readline()

                    # 处理最后一个缓存
                    # 抽取所有事件
                    events = Event.form_events(words_list, second_tags_list, sentences, index_pairs_list,speakers)
                    # 将事件加入事件集
                    true_events.append(events)

                    # 合并tag，构造模型抽取的训练集
                    predict_examples.append(
                        [currentWords, current_first_tags, current_pos_tags, sentences[0], index_pairs_list[0],
                         speakers[0]])


    extractor = Extractor()
    #单句单事实直接准确匹配
    events = extractor.extractor_from_words_posTags(predict_examples)

    fz = 0
    fm1 = 0
    fm2 = 0
    true_events_total = 0
    pre_events_total = 0
    for events1, events2 in zip(true_events, events):
        true_events_total += len(events1)
        pre_events_total += len(events2)
        the_fz, the_fm1,the_fm2 = evalutaion(events1, events2)
        fz += the_fz
        fm1 += the_fm1
        fm2 += the_fm2

    print('总共事件:' + str(true_events_total) + '\t' + '事件参数触发词总数：' + str(fm1))
    print('预测得到事件：' + str(pre_events_total) + '\t' + '预测事件参数触发词总数：' + str(fm2))
    print('预测结果:\n正确个数： '+str(fz)+'\n P: ' + str(fz / fm1)+'\n R:'+str(fz/fm2))

    print('检查事件是否相同')
def merge(tagsList):
    mergedTags = ['O' for _ in range(len(tagsList[0]))]
    for tags in tagsList:
        for index,tag in enumerate(tags):
            if(tag.find('_Trigger')!=-1):
                mergedTags[index] = tag[:-8]
    return mergedTags

def evalutaion(events1,events2):
    fm1 = 0
    fm2 = 0
    fz = 0
    if(events1!=None and (events2==None or len(events2)==0)):
        for event in events1:
            fm1 += event.get_score()
    else:
        for event in events1:
            if (event == None ):
                continue

            #应得分
            fm1 += event.get_score()

            #实际得分
            for otherEvent in events2:
                if(otherEvent == None):
                    continue
                if(event.type.value==otherEvent.type.value and event.trigger.value==otherEvent.trigger.value):
                    the_fz= event.compare(otherEvent)
                    fz+=the_fz
                    break
        for event in events2:
            fm2 += event.get_score()
    return fz,fm1,fm2



if __name__ == '__main__':
    run()
    sys.exit(0)