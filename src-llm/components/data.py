import pandas as pd

df = pd.read_csv("src-llm/data/train/wikisql_train.csv")

# print(df.head)
print(f"original shape: {df.shape}")

# filter dataframe where length of sql is less than 10
df = df[df['sql'].str.len() < 70]
print(f"filtered shape: {df.shape}")


# create prompt and target columns
with open("src-llm/data/train/raw_data.txt", "w") as f:
    for i, row in df.iterrows():
        prompt = f"question: {row['question']}\nsql:{row['sql']}"
        f.write(prompt + "\n")
        
    
    