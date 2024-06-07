import torch
from build_dataset import load_dataset
from gpt import GPT
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import *
from tokenizer import BPETokenizer
from tqdm import tqdm
import os


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = load_dataset()

tokenizer = BPETokenizer()
tokenizer.load('tokenizer.bin')
pad_ids,_ = tokenizer.encode(PAD)

def batch_proc(batch):
    bos_ids,_ = tokenizer.encode(BOS)
    eos_ids,_ = tokenizer.encode(EOS)
    pad_ids,_ = tokenizer.encode(PAD)

    batch_x = []
    batch_chatml = []
    # bpe encode
    for sample in batch:
        ids, chatml = sample
        ids = bos_ids + ids + eos_ids
        batch_x.append(ids)
        batch_chatml.append(chatml)

    # padding
    max_len = max([len(ids) for ids in batch_x])
    for ids in batch_x:
        if len(ids) < max_len:
            ids.extend(pad_ids * (max_len - len(ids)))
    batch_x = torch.tensor(batch_x, dtype=torch.long)

    # padding_mask
    batch_padding_mask = (batch_x == pad_ids[0])
    return batch_x, batch_padding_mask, batch_chatml

if __name__ == '__main__':

    model = GPT(d_model=GPT_DIM, nhead=GPT_HEAD, feedforward=GPT_FF, vocab_size=tokenizer.vocab_size(),seq_max_len=MAX_SEQ_LEN).to(DEVICE) # 模型
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99) # 优化器
    try:
        checkpoint = torch.load('checkpoint.bin')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        checkpoint = {'iter':0}

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8,shuffle=True, persistent_workers=True, collate_fn=batch_proc)

    pbar = tqdm(total=TRAIN_ITER, initial=checkpoint['iter'], postfix={'loss'})
    for i in range(checkpoint['iter'], TRAIN_ITER):
        batch_ids, batch_padding_mask, batch_chatml = next(iter(dataloader))

        batch_ids = batch_ids.to(DEVICE)
        batch_padding_mask = batch_padding_mask.to(DEVICE)

        logtis = model(batch_ids, batch_padding_mask) # (batch, seq, vocab)
        print(logtis.shape) # torch.Size([50, 228, 506])

        probs = logtis[:,:-1,:] # (batch, seq-1, vocab) 去掉最后一个终止符或者padding符号
        print(probs.shape) # torch.Size([50, 227, 506])
        targets = batch_ids[:,1:] # (batch, seq-1) 去掉最开始的开始符
        print(targets.shape) # torch.Size([50, 227])
        print(probs.size()) # torch.Size([50, 227, 506])
        # print(probs.view(-1, probs.size(2)))
        # print(targets.view(-1))
        print(probs.reshape(-1, probs.size(2)).shape) # torch.Size([11350, 506]) 拍平的操作之后的形状是 (batch*seq-1, vocab)
        print(targets.reshape(-1).shape) # torch.Size([11350]) 拍平的操作之后的形状是 (batch*seq-1)
        loss = F.cross_entropy(probs.reshape(-1, probs.size(2)), targets.reshape(-1), ignore_index=pad_ids[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(1)
        pbar.set_postfix({'loss':loss.item()})

        if i % 1000 == 0:
            checkpoint = {'iter': i, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, 'checkpoint.bin.tmp')
            os.replace('checkpoint.bin.tmp', 'checkpoint.bin')

    pbar.close()