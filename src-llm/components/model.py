from data_loader import create_dataloader
from multi_head import MultiHeadAttention
import torch.nn as nn
import torch
import logging

with open("notebooks/instructions.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257
output_dim = 256
max_len = 1024
block_size = max_len

token_embedding_layer = nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(block_size, output_dim)

max_length = 4
dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=5)

for batch in dataloader:
    x, y = batch

    token_embeddings = token_embedding_layer(x)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    input_embeddings = token_embeddings + pos_embeddings

    break
logging.info(f" {input_embeddings.shape}")
print(input_embeddings.shape)
torch.manual_seed(123)

# block_size = max_length
# d_in = output_dim
# d_out = d_in

# mha = MultiHeadAttention(d_in, d_out, block_size, 0.0, num_heads=2)

# batch = input_embeddings
# context_vecs = mha(batch)

# print("context_vecs.shape:", context_vecs.shape)