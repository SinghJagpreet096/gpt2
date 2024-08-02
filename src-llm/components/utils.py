import torch
from config import GPT_CONFIG_124M
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def get_batch(dataloader):
    for batch in dataloader:
        X, Y = batch
        return X, Y

eval_iters = GPT_CONFIG_124M["eval_iters"]
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

def processing_time(start_time, end_time):
    # convert to hours and minutes and  secocnds
    # Calculate time difference
    time_difference = end_time - start_time

    # Convert time difference to hours, minutes, and seconds
    hours = time_difference // 3600
    time_difference %= 3600
    minutes = time_difference // 60
    time_difference %= 60
    seconds = time_difference

    # Print the time taken
    return (f"Time taken: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds")

def get_bleu_score(true_query,generated_query):
    smoothing_function = SmoothingFunction().method1
    bleu_scores = sentence_bleu([true_query.split()], generated_query.split(),smoothing_function=smoothing_function) 
    return bleu_scores

def get_rouge_score(true_query,generated_query):
    rouge = Rouge()
    scores = rouge.get_scores(generated_query, true_query)
    rouge_score = scores[0]['rouge-l']['f']
    return rouge_score