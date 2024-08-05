## GPT FROM SCRATCH
The following is the code to train a Generative Pretrained Transformer model from scratch. Using the code you can train your own small chatbot similar to ChatGPT for your application or use case. 

Please note, it is only for learning purpose as training an actual LLM requires alot of compute and data.


How the run the code 
create virtual env
```bash 
$ pyenv virtualenv 3.10.12 <env name>

$ pyenv activate <env name>

$ pip install -r requirements.txt
```

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


Please note, it is only for learning purpose as training an actual LLM requires alot of compute and data. So, do your own research if planning to implement this in any production environment.

Please reach out if encounter any issues or want to contribute. I'd be happy to chat.

## References

[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)