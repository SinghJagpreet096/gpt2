# GPT FROM SCRATCH
This repository contains code to train a Generative Pretrained Transformer (GPT) model from scratch. Using this code, you can train your own small chatbot similar to ChatGPT for your application or use case.

Please note, this is for learning purposes only, as training a fully-fledged LLM requires substantial computational resources and data.


## Setup

### Create Virtual Environment
1. Create a virtual environment:
    ```sh
    pyenv virtualenv 3.10.12 <env_name>
    ```
2. Activate the virtual environment:
    ```sh
    pyenv activate <env_name>
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
## Pretraining

1. Create Data for pretraining
```bash
$ python src-llm/run.py -c data_prep
```

2. Pretraining a foundation model

```bash
$ python src-llm/run.py -c pretrain_and_save
```
3. if you want to just run the code and donot wish to save the model run the following cmd

```bash
$ python src-llm/run.py -c pretrain
```
4. Test the Pretrained model.
- update the model in app.py
```bash
$ streamlit run src-llm/app.py
```

### FineTunning

1. Create Data for pretraining
```bash
$ python src-llm/run.py -c data_prep_ft
```

2. Finetune the Pretrained model
```bash
$ python src-llm/run.py -c finetune_and_save
```
3. if you want to just run the code and donot wish to save the model run the following cmd

```bash
$ python src-llm/run.py -c finetune
```
4. Test the Pretrained model.
- update the model in app.py
```bash
$ streamlit run src-llm/app.py
```

## Important Note

This project is intended for learning purposes. Training a production-level LLM requires extensive computational power and data. Please conduct thorough research before implementing this in any production environment.

## Contributions

If you encounter any issues or wish to contribute, please reach out. I'd be happy to collaborate.

## References

[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)