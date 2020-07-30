
import numpy as np
from bert_serving.client import BertClient
import os
class query2BertVec:
    def __init__(self, model_path='multi_cased_L-12_H-768_A-12'):
        self.model_path = model_path
        print('please set -model_dir'+model_path+'')

    def textsEmbedding(self,text):
        if len(text)<0:
            text=['.\t']
        with BertClient(port=5555, port_out=5556) as bc:
         text_vecs =bc.encode(text)

        return text_vecs


