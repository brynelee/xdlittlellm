VOCAB_SIZE=500
MAX_SEQ_LEN=2000     # GPT模型输入限制

# transformer
GPT_DIM=384
GPT_HEAD=6
GPT_FF=1024
GPT_BLOCKS=6

# training
TRAIN_ITER=1000
BATCH_SIZE=50

# inference
TEMPERATURE = 1.2
TOP_K = 20

# special tokens
IM_START='<|im_start|>'
IM_END='<|im_end|>'
BOS='<|beginoftext|>'
EOS='<|endoftext|>'
PAD='<|padding|>'

# chat or generate
GPT_MODE='generate'
# GPT_MODE='chat'