import os

def run_model(epochs, model, loss, labels=None):
    command = f'python run_recbole.py --epochs={epochs} --model={model} --neg_sampling=None --log_wandb=True --use_gpu=True --loss_type={loss}'

    if labels is not None:
        command += f'--LABEL_FIELD={labels}'

    return command

epochs = 100
models = [
    'GRU4Rec', 
    'BERT4Rec',
    ]
loss = [
    'BPR', 
    'SupCon',
    ]
labels = [
    None,
    'skipped'
    ]

for model in models:
    for loss in loss:
        for label in labels:
            os.system(run_model(epochs, model, loss, labels=label))