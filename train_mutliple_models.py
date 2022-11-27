import os

def run_model(epochs, model, loss, labels):

    command = f'python run_recbole.py --epochs={epochs} --model="{model}" --log_wandb=True --use_gpu=True --loss_type="{loss}" '

    if loss == 'CE':
        command += "--train_neg_sample_args=None --neg_sampling=False "

        return command

    if labels is not None:
        command += f'--LABEL_FIELD="{labels}" --neg_sampling=False '


    return command

epochs = 100
models = [
    'GRU4Rec', 
    'BERT4Rec',
    ]
losses = [
    'CE'
    'BPR', 
    'SupCon',
    ]
labels = [
    None,
    'skipped'
    ]

for model in models:
    for loss in losses:
        for label in labels:
            break
            #os.system(run_model(epochs, model, loss, labels=label))


commands = [
    'python3 run_recbole.py --epochs=100 --model=GRU4Rec --neg_sampling=None --train_neg_sample_args=None --log_wandb=True --use_gpu=True --loss_type="CE" --dataset=lfm-100-no-skips',
    'python3 run_recbole.py --epochs=100 --model=DIEN --neg_sampling=None --train_neg_sample_args=None --log_wandb=True --use_gpu=True --dataset=lfm-100k',
    'python3 run_recbole.py --epochs=100 --model=DIN --neg_sampling=None --train_neg_sample_args=None --log_wandb=True --use_gpu=True --dataset=lfm-100k',
    'python3 run_recbole.py --epochs=100 --model=RepeatNet --neg_sampling=None --train_neg_sample_args=None --LABEL_FIELD=skipped --log_wandb=True --use_gpu=True --dataset="lfm-100k',

    ]
    

for command in commands:
    os.system(command)
