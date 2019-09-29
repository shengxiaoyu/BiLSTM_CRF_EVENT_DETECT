#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

from json import dumps

from flask import Flask, request, jsonify
from event_dectect.Event_Model.extract_event import Extractor
from conflict_detect.event_alignment.alignmenter import alignmenter
from conflict_detect.check_conflict.check_conflict import checker_conflict
from model import Relation,Event

app = Flask(__name__)
extractor = Extractor()
ali = alignmenter()
checker = checker_conflict()
@app.route('/api/event_extractor',methods=['POST'])
def predict():
    body = request.json
    print(body)
    if(not 'content' in body):
        return jsonify(code=3,message='json author error.')
    paragraphs = body['content']
    result = []
    if(isinstance(paragraphs,list)):
        for paragraph in paragraphs:
            events = extractor.extractor(paragraph)
            relations = relation_detect(events)
            result.append({'events':events,'relations':relations})
    elif(isinstance(paragraphs,str)):
        events = extractor.extractor(paragraphs)
        relations = relation_detect(events)
        result.append({'events': events, 'relations': relations})


    return jsonify(dumps(result,cls=MyEncoder))

def relation_detect(events):
    events_relations = []
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            if (ali.alignment(events[i], events[j])):
                if (checker.check(events[i], events[j])):
                    events_relations.append(Relation(None, 'Contradictory', events[i], events[j]))
                else:
                    events_relations.append(Relation(None, 'Entailment', events[i], events[j]))
    return events_relations
@app.route('/api/test',methods=['GET'])
def test():
    print("连接成功")

from json import JSONEncoder
class MyEncoder(JSONEncoder):
    def default(self, o):
        # return o.__dict__
        if isinstance(o, Event) or isinstance(o,Relation):
            return o._obj_to_json()
        return JSONEncoder.default(self,o)
if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8000)