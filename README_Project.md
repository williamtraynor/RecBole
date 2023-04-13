# MSciProject
Code Base for my MSci Project

To create env:
`conda env create -f environment.yml`

conda config --append channels conda-forge

`pip install -r requirements.txt`

`pip install -e . --verbose`
`pip install ray>0.7.5`
`pip install pandas, tabulate, torch`

To obtain the dataset, download the data in this folder:

`https://drive.google.com/drive/folders/1TuJ-wQBIVXeYTqWCIcyEOOnwBgmUoHkW`,

and place the files into the directory `dataset/music4all_1M`


To run a model use the command

`python run_recbole.py --model={MODEL_NAME} --dataset=Music4All-1M  --use_gpu=True {hyperparams}`,

where hyperparams can be found below for each of the baseline models.

BPR: `--embedding_size=2048 --learning_rate=0.0001 --train_batch_size=512`

LightGCN: `--embedding_size=512 --learning_rate=0.0005 --n_layers=1 --reg_weight=0.0001`

MultiVAE: No need to specify params

MacridVAE: `--dropout_prob=0.3 --embedding_size=128 --kafc=10 --learning_rate=0.001`

Diffusion (Ours (MultiVAE)): No need to specify params

MacridDiffusion (Ours (MacridVAE)): `--dropout_prob=0.3 --embedding_size=128 --kafc=3 --learning_rate=0.005`


To run the hyperparameter testing, run

`python3 run_hyper.py --{MODEL_NAME}_hyper_tuning.py`

The results of the hyperparameter testing will be written to the corresponding {MODEL_NAME}.txt file in `hyper_tuning/{MODEL_NAME}/{MODEL_NAME}.txt`