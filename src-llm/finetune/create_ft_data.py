import pandas as pd
import sys
import os



train = pd.read_parquet('data/train/train-00000-of-00001.parquet', columns=['question', 'query'])
test = pd.read_parquet('data/validation/validation-00000-of-00001.parquet', columns=['question', 'query'])

# print(data.head())
def main():

    def create_finetune_data(data, filename):
        with open(filename, "w") as f:
            for i, row in data.iterrows():
                f.write(f"USER: {row['question']}\n")
                f.write(f"AGENT: <|startoftext|>{row['query']}<|endoftext|>\n")
    os.makedirs("data/finetune", exist_ok=True)
    ## create training data
    create_finetune_data(train, "data/finetune/spider_train.txt")

    # create test data  
    create_finetune_data(test, "data/finetune/spider_test.txt")


