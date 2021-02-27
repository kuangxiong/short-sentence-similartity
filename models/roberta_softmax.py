# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: roberta_softmax.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Thu Feb 25 22:17:08 2021
---------------------------------------------------------
'''
import os 
import tensorflow as tf 
from tensorflow import keras 
from keras_bert import load_trained_model_from_checkpoint
from keras.layers import Lambda

import sys 
sys.path.append("..")

from config import GlobalData

class RobertaModelConfig(GlobalData):
    """
        RobertaModelConfig 模型配置参数
    """

    def __init__(self, modelname):
        super().__init__(modelname)
        self.num_epochs = 10
        self.max_len = 128 
        self.batch_size = 32 
        self.learning_rate = 0.00005
        self.nclass = 2
        # roberta模型配置文件
        #self.global_config = GlobalData(model_name)
        #self.bert_config_path = global_config.config_path 
        #self.bert_checkpoint_path = global_config.checkpoint_path
        #self.bert_vocab_path = global_config.vocab_path


def roberta_softmax(ModelConfig):
    """roberta_softmax.
        roberta_softmax 模型构建
    Parameters
    ----------
    ModelConfig :
        ModelConfig 模型配置参数
    """
    bert_model = load_trained_model_from_checkpoint(ModelConfig.config_path,
            ModelConfig.checkpoint_path, seq_len=None)
    for l in bert_model.layers:
        l.trainable = True
    text_id = tf.keras.layers.Input(shape=(ModelConfig.max_len,),
            dtype=tf.int32, name='text_id')
    segment_id = tf.keras.layers.Input(shape=(ModelConfig.max_len, ),
            dtype=tf.int32, name='segment')
    bert_output = bert_model([text_id, segment_id])
    first_bert_output = Lambda(lambda x: x[:,0])(bert_output)

    output = keras.layers.Dense(ModelConfig.nclass, activation='softmax')(first_bert_output)

    model = keras.Model(inputs=[text_id, segment_id], outputs=[output])
    return model

