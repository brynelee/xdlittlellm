from gpt import GPT
from config import *
import torch
from tokenizer import BPETokenizer
import torch.nn.functional as F
import random

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 分词器
tokenizer = BPETokenizer()
tokenizer.load('tokenizer.bin')

# 加载模型
model = GPT(d_model=GPT_DIM, nhead=GPT_HEAD, feedforward=GPT_FF, vocab_size=tokenizer.vocab_size(), seq_max_len=MAX_SEQ_LEN).to(DEVICE)
try:
    checkpoint = torch.load('checkpoint.bin')
    model.load_state_dict(checkpoint['model'])
except:
    pass

# 设置为评估模式，关闭dropout
model.eval()

# 可能的结束符
eos_ids,_ = tokenizer.encode(EOS)
pad_ids,_ = tokenizer.encode(PAD)
im_end_ids,_ = tokenizer.encode(IM_END)

def chat(query):
    global tokenizer, model

    inputs = f'{BOS}{IM_START}user\n{query}\n{IM_END}\n{IM_START}assistant\n' if GPT_MODE == 'chat' else f'{BOS}{query}'
    ids,_ = tokenizer.encode(inputs)

    while len(ids) < MAX_SEQ_LEN:
        batch_ids = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        print("batch_ids.shape: ", batch_ids.shape) # torch.Size([1, 4])
        batch_padding_mask = torch.tensor([[0]*len(ids)], dtype=torch.bool).to(DEVICE)
        print("batch padding_mask.shape: ", batch_padding_mask.shape) #  torch.Size([1, 4])

        with torch.no_grad():
            logits = model(batch_ids, batch_padding_mask) # (batch, seq, vocab)
            print("logits.shape: ", logits.shape) # torch.Size([1, 4, 506])
            # 多样性控制
            logits_updated = logits[0, -1, :]/TEMPERATURE # torch.Size([506])
            print("logits_updated.shape: ", logits_updated.shape)
            topk_logits, topk_ids = torch.topk(logits_updated, k=TOP_K)
            topk_logits, topk_ids = topk_logits.cpu(), topk_ids.cpu()
            # 从topk中随机选择一个token
            topk_probs = F.softmax(topk_logits, dim=-1)
            rnd = random.random()
            cumsum = 0
            for i in range(TOP_K):
                if rnd < cumsum + topk_probs[i]:
                    next_id = topk_ids[i].item()
                    break
                cumsum += topk_probs[i]

        if next_id in eos_ids+pad_ids+im_end_ids:
            break
        ids = ids + [next_id]

    return tokenizer.decode(ids[1:])

if __name__ == '__main__':
    while True:
        query = input('请输入：')
        if query == 'exit':
            break
        
        resp = chat(query)
        print('\n回复：<', resp)

