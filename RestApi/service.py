#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

from flask import Flask, request, jsonify
from json import dumps
from Event_Model.extract_event import Extractor
from Config.config_parser import getParser

app = Flask(__name__)
extractor = Extractor()
@app.route('/api/event_extractor',methods=['POST'])
def predict():
    body = request.json
    print(body)
    if(not 'content' in body):
        return jsonify(code=3,message='json author error.')
    # events = extractor2(body['content'][0])
    paragraphs = body['content']
    result = []
    if(isinstance(paragraphs,list)):
        for paragraph in paragraphs:
            result.append(extractor.extractor2(paragraph))
    elif(isinstance(paragraphs,str)):
        result.extend(extractor.extractor2(paragraphs))
    return jsonify(dumps(result,cls=MyEncoder))

from json import JSONEncoder
class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8000)