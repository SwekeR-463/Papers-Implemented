import torch

SRC_VOCAB_SIZE = 10000 # dummy 
TGT_VOCAB_SIZE = 10000 # dummy
EMBED_SIZE = 512
NUM_LAYERS = 6
HEADS = 8
HIDDEN_DIM = 2048
DROPOUT = 0.1
MAX_LEN = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"