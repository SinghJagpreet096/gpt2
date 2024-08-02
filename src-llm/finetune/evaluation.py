import torch
import sys
sys.path.append('src-llm/')
from components.gpt_model import GPTModel
from components.config import GPT_CONFIG_124M
import tiktoken


device = 'cuda' if torch.cuda.is_available() else 'cpu'
allowed_special = {"<|startoftext|>","<|endoftext|>"}
# load model
model = GPTModel(GPT_CONFIG_124M).to(device)
model.load_state_dict(torch.load('src-llm/artifacts/model_qa_2024-03-24_06-04-14.pt'))

optimizer = torch.optim.AdamW(model.parameters(), lr=GPT_CONFIG_124M["learning_rate"])

print("Model loaded")

tokenizer = tiktoken.get_encoding('gpt2')
prompt = f"USER:How many singers do we have?\nAGENT:<|startoftext|>"
context = torch.tensor(tokenizer.encode(prompt,allowed_special=allowed_special)).view(1, -1)
simple_generate = (tokenizer.decode(model.generate(context, max_new_tokens=1000)[0].tolist()))

query_generator = (tokenizer.decode(model.generate(context, max_new_tokens=1000,end_token = tokenizer.encode("<|endoftext|>",allowed_special=allowed_special))[0].tolist()))

print("Simple generate: \n", simple_generate)
print("Simple generate length: ", len(simple_generate)," \n\n")
print("Query generate: \n", query_generator)
print("Query generate length:", len(query_generator)," \n\n")