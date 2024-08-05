import torch
import pandas as pd
import sys
from components.data_loader import create_dataloader
from components.gpt_model import GPTModel
from components.config import GPT_CONFIG_124M
import logging
import tiktoken
import os
from datetime import datetime
from accelerate import Accelerator

import random
random.seed(0)
torch.manual_seed(0)
accelerator = Accelerator()

allowed_special = {"<|startoftext|>","<|endoftext|>"}
n_freeze = 6 # total layer is 12
batch_size = GPT_CONFIG_124M["batch_size"]
block_size = GPT_CONFIG_124M["ctx_len"]
learning_rate = GPT_CONFIG_124M["learning_rate"]
device = accelerator.device
eval_iters = GPT_CONFIG_124M["eval_iters"]
max_iters = GPT_CONFIG_124M["max_iters"]
eval_interval = GPT_CONFIG_124M["eval_interval"]

def main(save=False):
    def param_count(m):
        params = sum([p.numel() for p in m.parameters()])/1_000_000
        trainable_params = sum([p.numel() for p in m.parameters() if p.requires_grad])/1_000_000
        print(f"Total params: {params:.2f}M, Trainable: {trainable_params:.2f}M")
        return params, trainable_params

    def get_batch(dataloader):
        for batch in dataloader:
            X, Y = batch
            return X, Y
    #loss
    @torch.no_grad()
    def estimate_loss(model,train_data, val_data):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            if split == 'train':
                data = train_data
            else:
                data = val_data
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(data)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out



    with open ("src-llm/data/finetune/spider_train.txt", "r") as f:
        data = f.read()
    n = int(len(data)*0.8)
    train_data = data[:n]
    val_data = data[n:]
    train_data = create_dataloader(train_data, batch_size=8, max_length=4, stride=5)
    val_data = create_dataloader(val_data, batch_size=8, max_length=4, stride=5)

    # load the model
    model = GPTModel(GPT_CONFIG_124M).to(device)
    model.load_state_dict(torch.load('src-llm/artifacts/model_2024-08-04_14-58-06.pt',weights_only=True))
    logging.info("Model loaded")
    print("Model loaded")

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=GPT_CONFIG_124M["learning_rate"])

    model, optimizer, train_data, val_data= accelerator.prepare(
        model, optimizer, train_data, val_data 
    )

    print("Model parameters:")
    params, trainable_params = param_count(model)

    # freeze layers (disable gradients)
    for param in model.parameters(): param.requires_grad = False
    for param in model.out_head.parameters(): param.requires_grad = True
    for param in model.trf_block[n_freeze].parameters(): param.requires_grad = True

    print("Model parameters after freeze weights:")
    params, trainable_params = param_count(model)
    
    # training loop
    # model.train()
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model,train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # # sample a batch of data
        xb, yb = get_batch(train_data)
    #     # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()    
    def save_model():
        os.makedirs("src-llm/artifacts", exist_ok=True)
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save the model
        torch.save(model.state_dict(), f"src-llm/artifacts/model_qa_{current_datetime}.pt")
        logging.info(f"model saved as model_{current_datetime}.pt")
        print(f"model saved as model_{current_datetime}.pt")
    if save:
        save_model()

if __name__ == "__main__":
    main()
