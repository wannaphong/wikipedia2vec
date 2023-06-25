# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

from __future__ import unicode_literals
import jieba
import logging
import re
import six
from pythainlp.tokenizer import word_tokenize

from .base_tokenizer cimport BaseTokenizer

def get_char_span(input_txt):
   doc = word_tokenize(input_txt)
   _list_data=[]
   idx=0
   for i, token in enumerate(doc):
    start_i = idx
    end_i = start_i + len(token)
    idx+=len(token)
    _list_data.append((token,start_i,end_i))
   return _list_data

cdef class PyThaiNLPTokenizer(BaseTokenizer):
    cdef _rule

    def __init__(self):
        self._rule = re.compile(r'^\s*$')

    cdef list _span_tokenize(self, unicode text):
        return [(start, end) for (word, start, end) in get_char_span(text)
                if not self._rule.match(word)]

    def __reduce__(self):
        return (self.__class__, tuple())
