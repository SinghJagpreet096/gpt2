# import requests
# import os



# API_TOKEN = os.getenv("HF_TOKEN")
# headers = {"Authorization": f"Bearer {API_TOKEN}"}
# API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b"
# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()
# data = query({"inputs": "The answer to the universe is"})
# print(data)
# !pip3 install -q -U bitsandbytes==0.42.0
# !pip3 install -q -U peft==0.8.2
# !pip3 install -q -U trl==0.7.10
# !pip3 install -q -U accelerate==0.27.1
# !pip3 install -q -U datasets==2.17.0
# !pip3 install -q -U transformers==4.38.0
from transformers import AutoModelForCausalLM,AutoTokenizer
import os

from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
import torch
import os

class EndpointHandler():

    def __init__(self, model_id=""):
        self.device = "cpu:0"
        # self.bnb_config = BitsAndBytesConfig(load_in_4bit=True,
        #                                      bnb_4bit_quant_type="nf4",
        #                                      bnb_4bit_compute_dtype=torch.bfloat16,)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id,
                                                          device_map={"":0},
                                                        #   quantization_config=self.bnb_config,
                                                        )
        
    def __call__(self, input:str) -> str:
        
        inputs = self.tokenizer(input, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=20)
        result = (self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        return result
model_id = "NumbersStation/nsql-350M"   
handler = EndpointHandler(model_id)
text = """CREATE TABLE stadium (
    stadium_id number,
    location text,
    name text,
    capacity number,
    highest number,
    lowest number,
    average number
)

CREATE TABLE singer (
    singer_id number,
    name text,
    country text,
    song_name text,
    song_release_year text,
    age number,
    is_male others
)

CREATE TABLE concert (
    concert_id number,
    concert_name text,
    theme text,
    stadium_id text,
    year text
)

CREATE TABLE singer_in_concert (
    concert_id number,
    singer_id text
)

-- Using valid SQLite, answer the following questions for the tables provided above.

-- {query}

SELECT
"""
(handler(text))