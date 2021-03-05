import tensorflow as tf 
from tensorflow import keras 
import numpy as np
from keras_bert import get_custom_objects
import time
import os
import re
import csv

from config import GlobalData 
from models.roberta_softmax import RobertaModelConfig, roberta_softmax


if __name__=='__main__':
    modelconfig = RobertaModelConfig("roberta_wwm")
    
    start_time = time.time()
    save_model = os.path.join("./data/save_model", "roberta_softmaxbert_model.h5")
    model = roberta_softmax(modelconfig) 
    model = keras.models.load_model(save_model,
            custom_objects=get_custom_objects())
    print(model.summary())

    res  = []
    all_inputs_index, all_inputs_seg = [], []
    with open(modelconfig.test_path, "r") as reader:
        for lines in reader:
            tmplist = re.split('[\t\n]', lines)
            tmp1 = list(map(int, tmplist[0].split()))
            for i in range(len(tmp1)):
                if tmp1[i] > 21120:
                    tmp1[i] = 100
            tmp2 = list(map(int, tmplist[1].split()))
            for i in range(len(tmp2)):
                if tmp2[i] > 21120:
                    tmp2[i] = 100
            tmp_ind = [101] + tmp1 + [102] + tmp2 +[102]
            tmp_seg = [0] * (len(tmp1)+1) + [1]*(len(tmp2)+1)
            inputs_ind = np.array(tmp_ind + [0]*(modelconfig.max_len -
                        len(tmp_ind)))
            inputs_seg = np.array(tmp_seg + [0]*(modelconfig.max_len -
                        len(tmp_seg)))
            predict_val = model.predict([np.array([inputs_ind]),
                    np.array([inputs_seg])])
            print(predict_val[0])
            res.append(predict_val[0])
    f_name = open("result.csv", 'w')
    f_writer = csv.writer(f_name)
    for e in res:
        f_writer.writerow(e)
    f_name.close()

