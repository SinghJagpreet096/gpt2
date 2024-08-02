import os
import shutil
from tqdm import tqdm
import pandas as pd
import time
from datasets import load_dataset
from special_token import SpecialToken


def main():
    start_time = time.time()
    data_path = "Clinton/Text-to-sql-v1"
    dataset = load_dataset(data_path,)
    
    print(f"length of sample: {len(dataset['train']):,}") 
    def create_promt(data):
        instruction = data["instruction"]
        context = data["input"]
        response = data["response"]
        start_token = SpecialToken.start_token
        end_token = SpecialToken.end_token
        template = "Below are sql tables schemas paired with instruction that describes a task.\n Using valid SQLite, write a response that appropriately completes the request for the provided tables."
        text = f"{template}\n### {instruction}\n### {context}\n### {start_token+response+end_token}"
        return text  
    shutil.rmtree("src-llm/data/train")
    os.makedirs("src-llm/data/train")
    with open("src-llm/data/train/sql_context.txt", "w") as f:

        for data in dataset["train"]:
            f.write(create_promt(data) + "\n") 

    end_time = time.time()
    # print(f"Processing time: {end_time-start_time:.2f} seconds")
    # processing_time(start_time,end_time)

# if __name__ == "__main__":
#     main()