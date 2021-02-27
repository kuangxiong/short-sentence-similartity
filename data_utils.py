# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: data_utils.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Fri Feb 26 12:46:04 2021
---------------------------------------------------------
'''
import codecs
import tensorflow as tf 
from tensorflow import keras
from keras_bert import Tokenizer
from config import GlobalData
from models.roberta_softmax import RobertaModelConfig
import numpy as np

class BertDataPreload(object):
    ""
    ""
    def __init__(self, ModelConfig):
        """__init__.

        Parameters
        ----------
        ModelConfig :
            ModelConfig 模型配置数据
        """
        self.modelconfig = ModelConfig 

    def load_data(self):
        """
            download training data and test data
        """

        # training data \t testing data
        train_data, test_data = [], []
        with codecs.open(self.modelconfig.train_path, "r", "utf-8") as reader:
            for line in reader:
                segment = line.strip()
                tmp_list = segment.split()
                if len(tmp_list)==3:
                    train_data.append(segment.split())

        with codecs.open(self.modelconfig.test_path, "r", "utf-8") as reader:
            for line in reader:
                segment = line.strip()
                tmp_list = segment.split()
                if len(tmp_list)==2:
                    test_data.append(segment.split())

        return train_data, test_data
                
        
    
    def bert_text2id(self, train_text, training=True):
        """bert_text2id.
           将字转化为字id
        Parameters
        ----------
        train_text :
            train_text 文本数据
        training :
            training 数据是否是用于训练
        """
        token_dict = {}
        with codecs.open(self.modelconfig.vocab_path, "r", "utf-8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)

        tokenizer = Tokenizer(token_dict)
        data_X_ind, data_X_seg = [], []
        data_Y = []
        N = len(train_text)
        for i in range(N):
            if training==True:
                seg1, seg2, label = train_text[i][0], train_text[i][1], train_text[i][2]
                data_Y.append(label)
            else:
                seg1, seg2 = train_text[i][0], train_text[i][1]

            indices, segments = tokenizer.encode(first=seg1, second=seg2,
                    max_len = self.modelconfig.max_len)
            data_X_ind.append(np.array(indices))
            data_X_seg.append(np.array(segments))
        data_X_ind = keras.preprocessing.sequence.pad_sequences(
                data_X_ind, 
                maxlen = self.modelconfig.max_len,
                dtype="int32",
                padding="post",
                truncating="post",
                value = 0.0
                )

        data_X_seg = keras.preprocessing.sequence.pad_sequences(
                data_X_seg, 
                maxlen = self.modelconfig.max_len,
                dtype="int32",
                padding="post",
                truncating="post",
                value=0
                )
        if training == True:
            return data_X_ind, data_X_seg, data_Y
        else:
            return data_X_ind, data_X_seg

            
if __name__=='__main__':

    config = RobertaModelConfig("bert-wwt-ext")
    print(config)
    BertData = BertDataPreload(config)
    train_data, test_data, rest = BertData.load_data()
    data_X_ind, data_X_seg, data_Y = BertData.bert_text2id(train_data)
    print(rest)
    print(data_X_ind[0])
    
   # print(train_data)
    
