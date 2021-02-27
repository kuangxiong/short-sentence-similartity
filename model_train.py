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
from keras.utils.np_utils import to_categorical



if __name__=="__main__":
    model_name = "roberta_softmax"
    ModelConfig = myconfig('roberta_softmax')
    BertData = BertDataPreload(ModelConfig)

    train_data, test_data = BertData.load_data()
    data_X_ind, data_X_seg, int_Y = BertData.bert_text2id(train_data)
    # int_Y 转换成one-hot 编码
    data_Y = to_categorical(int_Y, num_classes=None)
    print(data_Y[1])
    
    model = mymodel(ModelConfig)
    adam = tf.keras.optimizers.Adam(ModelConfig.learning_rate)
    model.compile(
            loss = "categorical_crossentropy",
            optimizer=adam,
            metrics=['accuracy']
            )
    save_path = ModelConfig.save_model
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
            f"{save_path}bert_model.h5", save_best_only=True
            )
    early_stopping_cb = keras.callbacks.EarlyStopping(
            patience=4, 
            restore_best_weights=True
        )
    train_X_test, train_X_seg = np.asarray(data_X_ind), np.asarray(data_X_seg)
    train_label = np.asarray(data_Y)
    history = model.fit(
            [train_X_test, train_X_seg],
            train_label, 
            epochs=50,
            batch_size = ModelConfig.batch_size,
            validation_split=0.2,
            validation_freq=1,
            callbacks=[checkpoint_cb, early_stopping_cb]
        )
    print(model.summary())
    print(data_X_ind[0])
    
