# -*- coding:utf-8 -*-
'''
---------------------------------------------------------
 File Name: config.py
 Author:kuangxiong
 Mail: kuangxiong@lsec.cc.ac.cn
 Created Time: Thu Feb 25 21:57:50 2021
---------------------------------------------------------
'''
import os 

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

class  GlobalData(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.train_path = os.path.join(BASE_PATH,
                'data/data_source/gaiic_track3_round1_train_20210228.tsv')
        self.test_path = os.path.join(BASE_PATH,
                'data/data_source/gaiic_track3_round1_testA_20210228.tsv')

        model_path = os.path.join(BASE_PATH,
                "data/model_source/chinese_roberta_wwm_ext_L-12_H-768_A-12")
        self.config_path = os.path.join(model_path, 'bert_config.json')
        self.vocab_path = os.path.join(model_path, 'vocab.txt')
        self.checkpoint_path = os.path.join(model_path, "bert_model.ckpt")
        self.save_model = os.path.join(BASE_PATH, f"data/save_model/{model_name}")

