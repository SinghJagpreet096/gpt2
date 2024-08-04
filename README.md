## GPT FROM SCRATCH
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


