import os
import shutil
from tqdm import tqdm
import time
from datasets import load_dataset
from components.utils import processing_time

def main():
    start_time = time.time()
    dataset = load_dataset("Bingsu/openwebtext_20p",)
    n = int(len(dataset["train"]["text"])*0.2)
    sample = dataset["train"]["text"][:20000]
    print(f"length of sample: {len(sample):,}")

    shutil.rmtree("src-llm/data/train")
    os.makedirs("src-llm/data/train")

    with open("src-llm/data/train/webtext-20p.txt", "w") as f:
        for s in sample:
            f.write(s + "\n")        
        
    end_time = time.time()
    # processing_time(start_time,end_time)
