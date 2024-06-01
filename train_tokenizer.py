
from tokenizer import BPETokenizer
from config import *
import os
import sys
import json

if os.path.exists("tokenizer.bin"):
    print("Tokenizer already exists")
    sys.exit(0)

# 原始数据
with open('纳兰性德诗集.json', 'r', encoding='utf-8') as fp:
    ds = json.loads(fp.read())

text_list = []
sample_count = 0
for sample in ds:
    text_list.append(sample['title'])
    for p in sample['para']:
        text_list.append(p)
    sample_count+=1
print('共加载%d条数据' % sample_count)

# 训练词表
tokenizer = BPETokenizer()
tokenizer.train(text_list, VOCAB_SIZE)
tokenizer.add_special_tokens([IM_START,IM_END,BOS,EOS,PAD])
tokenizer.save('tokenizer.bin')

# 打印词典
print('vocab:', tokenizer.vocab())

