import pandas as pd
import os
from datasets import load_dataset

dataset = load_dataset("xlangai/spider")

train = pd.DataFrame({'question': dataset['train']['question'], 'query': dataset['train']['query']})
test = pd.DataFrame({'question': dataset['validation']['question'], 'query': dataset['validation']['query']})

def create_finetune_data(data, filename):
        with open(filename, "w") as f:
            for i, row in data.iterrows():
                f.write(f"USER: {row['question']}\n")
                f.write(f"AGENT: <|startoftext|>{row['query']}<|endoftext|>\n")
# print(data.head())
def main():
    os.makedirs("src-llm/data/finetune", exist_ok=True)
    ## create training data
    create_finetune_data(train, "src-llm/data/finetune/spider_train.txt")
    # create test data  
    create_finetune_data(test, "src-llm/data/finetune/spider_test.txt")


