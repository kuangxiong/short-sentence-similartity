# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: model_train.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Thu Feb 25 22:09:49 2021
---------------------------------------------------------
'''

import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os 
import argparse
from data_utils import BertDataPreload

from models.roberta_softmax import roberta_softmax as mymodel 
from  models.roberta_softmax import RobertaModelConfig as myconfig 
from keras_bert import tokenizer



if __name__=="__main__":
    model_name = "roberta_softmax"
    modelconfig = myconfig('roberta_softmax')
    BertData = BertDataPreload(modelconfig)

    train_data, test_data = BertData.load_data()
    data_X_ind, data_X_seg, data_Y = BertData.bert_text2id(train_data)
    
    model = mymodel(modelconfig)
    print(model.summary())
    print(data_X_ind[0])
    
