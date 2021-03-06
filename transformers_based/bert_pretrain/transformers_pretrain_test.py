from transformers import BertTokenizer, BertModel
import tensorflow as tf 
import numpy as np

tokenizer=BertTokenizer.from_pretrained("model_source/checkpoint-70000")
model = BertModel.from_pretrained("model_source/checkpoint-70000")
inputs = tokenizer.encode("1 2 3 4 5 6", max_length=20, padding='max_length')
print(inputs)

#tmp = tokenizer.convert_ids_to_tokens(inputs)
#print(tmp)
#inputs = np.array([inputs])
#outputs = model(inputs)
#print(outputs)
