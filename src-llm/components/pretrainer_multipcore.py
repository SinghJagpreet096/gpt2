import logging
from components.gpt_model import GPTModel
from components.config import GPT_CONFIG_124M
from components.data_loader import create_dataloader
from accelerate import Accelerator
from torch.optim import AdamW
import torch
import os
import tqdm
from datetime import datetime


logger = logging.getLogger(__name__)
accelerator = Accelerator()

batch_size = GPT_CONFIG_124M["batch_size"]
block_size = GPT_CONFIG_124M["ctx_len"]
learning_rate = GPT_CONFIG_124M["learning_rate"]
eval_iters = GPT_CONFIG_124M["eval_iters"]
max_iters = GPT_CONFIG_124M["max_iters"]
eval_interval = GPT_CONFIG_124M["eval_interval"]
device = accelerator.device

def main(save=False):
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
    data_path = "src-llm/data/train/sql_context.txt"
    with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
    print(f"length of text: {len(text):,}")
    # text = text[:1000]
    sample = int(len(text)*0.2)
    text = text[:sample]
    print(f"length of sample: {len(text):,}")
    n = int(len(text)*0.9)
    train_data = text[:n]
    val_data = text[:n]
    train_data = create_dataloader(train_data, batch_size=8, max_length=4, stride=5)
    val_data = create_dataloader(val_data, batch_size=8, max_length=4, stride=5)

    print(f"data created")
    model = GPTModel(GPT_CONFIG_124M).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model, optimizer, train_data, val_data= accelerator.prepare(
        model, optimizer, train_data, val_data 
    )
   
    print(f"training begins")
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model,train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # # sample a batch of data
        xb, yb = get_batch(train_data)
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()

    def save_model():
        os.makedirs("src-llm/artifacts", exist_ok=True)
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save the model
        torch.save(model.state_dict(), f"src-llm/artifacts/model_{current_datetime}.pt")
        logging.info(f"model saved as model_{current_datetime}.pt")
        print(f"model saved as model_{current_datetime}.pt")
    if save:
        save_model()
    

if __name__ == "__main__":
    main()