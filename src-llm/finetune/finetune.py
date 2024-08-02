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
n_freeze = 8 # total layer is 12
batch_size = GPT_CONFIG_124M["batch_size"]
block_size = GPT_CONFIG_124M["ctx_len"]
learning_rate = GPT_CONFIG_124M["learning_rate"]
device = accelerator.device
eval_iters = 200
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



    with open ("data/finetune/spider_train.txt", "r") as f:
        data = f.read()
    n = int(len(data)*0.8)
    train_data = data[:n]
    val_data = data[n:]
    train_data = create_dataloader(train_data, batch_size=8, max_length=4, stride=5)
    val_data = create_dataloader(val_data, batch_size=8, max_length=4, stride=5)

    # load the model
    model = GPTModel(GPT_CONFIG_124M).to(device)
    model.load_state_dict(torch.load('artifacts/model_2024-03-23_01-39-17.pt'))
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
    for iter in range(10000):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model,train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # # sample a batch of data
        xb, yb = get_batch(train_data)
    #     # print(type(xb), type(yb))

    #     # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        # genarate a sequence of tokens
    # tokenizer = tiktoken.get_encoding('gpt2')
    # prompt = f"USER: How many singers do we have?\nAGENT: <|startoftext|>"
    # context = torch.tensor(tokenizer.encode(prompt,allowed_special=allowed_special)).view(1, -1)
    # simple_generate = (tokenizer.decode(model.generate(context, max_new_tokens=100)[0].tolist()))
    # print("\n\n")
    # # query_generator = (tokenizer.decode(model.generate_query(context, max_new_tokens=100,end_token = tokenizer.encode("<|endoftext|>",allowed_special=allowed_special))[0].tolist()))

    # print("Simple generate: \n", simple_generate)
    # print("Simple generate length: ", len(simple_generate)," \n\n")
    # print("Query generate: \n", query_generator)
    # print("Query generate length:", len(query_generator)," \n\n")
    
    def save_model():
        os.makedirs("src-llm/artifacts", exist_ok=True)
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save the model
        torch.save(model.state_dict(), f"artifacts/model_qa_{current_datetime}.pt")
        logging.info(f"model saved as model_{current_datetime}.pt")
        print(f"model saved as model_{current_datetime}.pt")
    if save:
        save_model()

if __name__ == "__main__":
    main()
