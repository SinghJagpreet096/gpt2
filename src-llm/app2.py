from transformers import AutoTokenizer, AutoModelForCausalLM
import chainlit as cl
import re
from langchain.prompts import ChatPromptTemplate
tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-350M")
model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-350M")



# PROMT = """CREATE TABLE stadium (
#     stadium_id number,
#     location text,
#     name text,
#     capacity number,
#     highest number,
#     lowest number,
#     average number
# )

# CREATE TABLE singer (
#     singer_id number,
#     name text,
#     country text,
#     song_name text,
#     song_release_year text,
#     age number,
#     is_male others
# )

# CREATE TABLE concert (
#     concert_id number,
#     concert_name text,
#     theme text,
#     stadium_id text,
#     year text
# )

# CREATE TABLE singer_in_concert (
#     concert_id number,
#     singer_id text
# )

# -- Using valid SQLite, answer the following questions for the tables provided above.

# -- What is the maximum, the average, and the minimum capacity of stadiums ?

# SELECT"""


# chat_template = ChatPromptTemplate.from_messages([
#     ("human", "What is the capital of {country}?"),
#     ("ai", "The capital of {country} is {capital}.")
# ])

# messages = chat_template.format_messages(country="Canada", capital="Ottawa")
# print(messages)

@cl.on_message
async def main(message: cl.Message):
    query = message.content
    PROMT = f"""CREATE TABLE stadium (
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

SELECT"""

    
    
    input_ids = tokenizer(PROMT, return_tensors="pt").input_ids

    generated_ids = model.generate(input_ids, max_length=500)
    pred = (tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    # Send a response back to the user
    if "SELECT" in pred:
        pred = pred.split("SELECT")[1]
    pred = "SELECT " + re.sub('"""', "",pred)
    
    await cl.Message(
        content=pred,
    ).send()