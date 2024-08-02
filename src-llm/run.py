import click
from components import pretrainer_multipcore
from components import prepare_data_openweb
from finetune import finetune
# from finetune import create_ft_data
from components import sql_create_context
# from finetune import create_ft_data
import time
from components.utils import processing_time

DATA_PREP = "data_prep"
DATA_PREP_FT = "data_prep_ft"
PRETRAIN = "pretrain"
PRETRAIN_SAVE = "pretrain_and_save"
FINE_TUNE = "fine_tune"
FINE_TUNE_SAVE = "fine_tune_and_save"  


@click.command()
@click.option(
    '--config',
    '-c',
    type=click.Choice([DATA_PREP,DATA_PREP_FT,PRETRAIN,PRETRAIN_SAVE, FINE_TUNE, FINE_TUNE_SAVE]), 
    help="Choose the pipe to run"
    "data_prep: Prepare data"
    "pretrain: Pretrain the model"
    "fine-tune: Fine-tune the model")

def run_pipelines(config:str):
    start_time = time.time()
    if config == DATA_PREP:
        sql_create_context.main()
    # elif config == DATA_PREP_FT:
    #     create_ft_data.main()
    elif config == PRETRAIN:
        pretrainer_multipcore.main()
    elif config == PRETRAIN_SAVE:
        pretrainer_multipcore.main(save=True)
    elif config == FINE_TUNE:
        finetune.main()
    elif config == FINE_TUNE_SAVE:
        finetune.main(save=True)
    else:
        raise ValueError(f"Invalid config: {config}")
    end_time = time.time()
    print(f"Pipeline {config} completed")
    print(f'{processing_time(start_time,end_time)}')  
    exit(0) 

if __name__ == "__main__":
    run_pipelines() 
   
    