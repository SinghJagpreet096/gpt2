from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from datasets import load_dataset
import sys
import time
import tqdm
sys.path.append('src-llm/components')
from utils import get_bleu_score,get_rouge_score,processing_time

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-350M")
model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-350M")
data = load_dataset('b-mc2/sql-create-context')
# print(data['train'])
data = data['train']
# print(data)
# print(data['question'])
data = pd.DataFrame(data) 
data['context'] = data['context'].apply(lambda x: re.sub(r";", '\n\n', x))
data = data[:int(len(data)*.10)]
n = int(len(data)*.8)
train_data = data[:n]
valid_data = data[n:]
train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
print("length of train_data: ", len(train_data))
print("length of valid_data: ", len(valid_data))

def create_promt(data):
    context = data["context"]
    question = data["question"]
    answer = data["answer"]
    # start_token = SpecialToken.start_token
    # end_token = SpecialToken.end_token
    template = f"""{context}

-- Using valid SQLite, answer the following questions for the tables provided above.

-- {question}

SELECT"""
        
    return template 

def get_prediction(text):
    promt = create_promt(text)
    # print(PROMT)

    input_ids = tokenizer(promt, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=500)
    pred = (tokenizer.decode(generated_ids[0], skip_special_tokens=True))
        # Send a response back to the user
    # print(pred)

    if "SELECT" in pred:
        q = 'SELECT'.join(pred.split("SELECT")[1:])
        result = ("\n\nSELECT"+q)
    return result

def evaluate(data):
    bleu_scores = []
    rouge_scores = []
    def average(lst):
        return sum(lst) / len(lst)
    for i in tqdm.tqdm(range(len(data))):
        text = data.loc[i]
        pred = get_prediction(text)
        true_query = data.loc[i]['answer']
        generatedd_query = pred.strip()
        bleu_score = get_bleu_score(true_query,generatedd_query)
        rouge_score = get_rouge_score(true_query,generatedd_query)
        bleu_scores.append(bleu_score)
        rouge_scores.append(rouge_score)
    return average(bleu_scores),average(rouge_scores)

bleu_score_train,rouge_score_train= evaluate(train_data)
print('bleu_score training data:',bleu_score_train)
print('rouge_score training data:',rouge_score_train)

bleu_score_valid,rouge_score_valid= evaluate(valid_data)
print('bleu_score validation data:',bleu_score_valid)
print('rouge_score validation data:',rouge_score_valid)

with open("src-llm/artifacts/results.txt", "w") as f:
    f.write(f"bleu_score training data: {bleu_score_train}\n")
    f.write(f"rouge_score training data: {rouge_score_train}\n")
    f.write(f"bleu_score validation data: {bleu_score_valid}\n")
    f.write(f"rouge_score validation data: {rouge_score_valid}\n")
end_time = time.time()
print(f'{processing_time(start_time,end_time)}')




