from transformers import BertConfig, BertForMaskedLM, BertTokenizer, \
     LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras
import tqdm
import random
import numpy as np
import os



train = pd.read_csv('data_source/gaiic_track3_round1_train_20210228.tsv',sep='\t', names=['text_a', 'text_b', 'label'])
test = pd.read_csv('data_source/gaiic_track3_round1_testA_20210228.tsv',sep='\t', names=['text_a', 'text_b', 'label'])
test['label'] = 0


##训练集和测试集造字典
from collections import defaultdict
def get_dict(data):
    words_dict = defaultdict(int)
    for i in range(data.shape[0]):
        text = data.text_a.iloc[i].split() + data.text_b.iloc[i].split()
        for c in text:
            words_dict[c] += 1
    return words_dict
test_dict = get_dict(test)
train_dict = get_dict(train)
word_dict = list(test_dict.keys()) + list(train_dict.keys())
word_dict = set(word_dict)
word_dict = set(map(int, word_dict))
word_dict = list(word_dict)
special_tokens = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
WORDS = special_tokens + word_dict
pd.Series(WORDS).to_csv('Bert-vocab.txt', header=False,index=0)

# tokenizer = BertWordPieceTokenizer("Bert-vocab.txt", lowercase=False, handle_chinese_chars=False)
tokenizer = BertTokenizer("Bert-vocab.txt", max_len=100, lowercase=False, handle_chinese_chars=False)
tokenizer.save_pretrained('mynewmodel')

"""
config = BertConfig(
    vocab_size=len(WORDS)+1,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model = BertForMaskedLM(config=config)

res = []
for i in range(len(train)):
  if len(train.iloc[i]['text_a'])>1:
    res.append(train.iloc[i]['text_a'])
  if len(train.iloc[i]['text_b'])>1:
    res.append(train.iloc[i]['text_b'])


for i in range(len(test)):
  if len(test.iloc[i]['text_a'])>1:
    res.append(test.iloc[i]['text_a'])
  if len(test.iloc[i]['text_b'])>1:
    res.append(test.iloc[i]['text_b'])

pd.Series(res).to_csv('sentence.txt', header=False,index=0)

dataset = LineByLineTextDataset(
    # 'bert_vocab',
    tokenizer=tokenizer,
    file_path="./sentence.txt",
    block_size=100,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
training_args = TrainingArguments(
    output_dir="./mynewmodel",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=1000,
    save_total_limit=1,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    # prediction_loss_only=True
)

trainer.train()

trainer.save_model("./mynewmodel")
"""
