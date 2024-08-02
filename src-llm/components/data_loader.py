import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging

class GPTDatset(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        try:
            self.tokenizer = tokenizer
            self.input_ids = []
            self.target_ids = []

            ## tokenize the entire text
            token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

            ## use a sliding window to chunk the book into overlapping sequence of max_length
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk =token_ids[i:i + max_length]
                target_chunk = token_ids[i + 1: i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))
        except Exception as e:
            logging.error(f"create data failed {e}")
            raise e

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, batch_size=4, max_length=256, stride=128,shuffle=True):
    # intialize tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')

    # create dataset
    dataset = GPTDatset(txt, tokenizer, max_length, stride)

    # create data loader
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=shuffle,)

    return dataloader

if __name__ == "__main__":
    with open("notebooks/instructions.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_text = tokenizer.encode(raw_text,allowed_special={'<|endoftext|>'})

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

    print(input_embeddings.shape)
