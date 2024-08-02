import tiktoken
import torch
from components.config import GPT_CONFIG_124M
from components.gpt_model import GPTModel
from torch.optim import AdamW
import logging
from components.data_loader import create_dataloader
from datetime import datetime
import time
import random
random.seed(42)
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
from datasets import load_dataset

batch_size = GPT_CONFIG_124M["batch_size"]
block_size = GPT_CONFIG_124M["ctx_len"]
learning_rate = GPT_CONFIG_124M["learning_rate"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = GPT_CONFIG_124M["eval_iters"]
max_iters = GPT_CONFIG_124M["max_iters"]
eval_interval = GPT_CONFIG_124M["eval_interval"]


# data loading
def get_batch(dataloader):
    for batch in dataloader:
        X, Y = batch
        return X.to(device), Y.to(device)


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


if __name__ == "__main__":

    with open("src-llm/data/train/webtext-20p.txt", "r", encoding="utf-8") as f:
        text = f.read()
    text = text[100:1000]
    # from datasets import load_dataset


    # dataset = load_dataset("Bingsu/openwebtext_20p", )


    # print(dataset['train']['text'][0:10])
    # text = dataset['train']['text']
    no_of_tokens = len(text)
    logging.info(f"no. of tokens: {no_of_tokens}")
    print(f"no of tokens OpenWeb-20p: {no_of_tokens}")
    # sample_text = text[:int(no_of_tokens/3)]
    # print(f"no of tokens in sample: {int(no_of_tokens/3)}")
    
    n = int(len(text) * 0.9)
    train_data = create_dataloader(text[:n], batch_size=8, max_length=4, stride=5)
    val_data = create_dataloader(text[n:], batch_size=8, max_length=4, stride=5)
    # print(f"train_data: {train_data.shape}, val_data: {val_data.shape}")

    # for batch in train_data:
    #     print(batch)
    #     break

    # # intialize model
    model = GPTModel(GPT_CONFIG_124M).to(device)

    # # initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # training loop
    start_time = time.time()
    print(f"training begins")
    for iter in range(max_iters):
        
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
        loss.backward()
        optimizer.step()
    # genarate a sequence of tokens
    tokenizer = tiktoken.get_encoding('gpt2')
    # prompt = "s"
    # context = torch.tensor(tokenizer.encode(prompt, allowed_special={"<|startoftext|>","<|endoftext|>"})).view(1, -1)
    # print(tokenizer.decode(model.generate(context, max_new_tokens=100)[0].tolist()))

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context, max_new_tokens=1000)[0].tolist()))

    # create directory to save the model
    os.makedirs("src-llm/artifacts", exist_ok=True)
    # save the model
    # torch.save(model.state_dict(), f"src-llm/artifacts/model_{current_datetime}.pt")
    logging.info(f"model saved as model_{current_datetime}.pt")
    logging.info(f"total time: {time.time() - start_time}")
    end_time = time.time()
    training_time = end_time - start_time
    hours, minutes, seconds = int(training_time // 3600), int((training_time % 3600) // 60), int(training_time % 60)
    # Display the training time
    print(f"Training time: {hours} hours, {minutes} minutes, {seconds} seconds")
