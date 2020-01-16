#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

__doc__ = 'description'
__author__ = '13314409603@163.com'

from RestApi import service

if __name__ == '__main__':
    service.app.run(host='0.0.0.0', port=8000,debug=False)