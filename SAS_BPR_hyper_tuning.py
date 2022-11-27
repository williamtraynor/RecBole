# -*- coding: utf-8 -*-
# @Time   : 2022/7/15
# @Author : Gaowei Zhang
# @Email  : zgw15630559577@163.com
import os
import unittest
from recbole.trainer import HyperTuning
from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.utils import (
    get_model,
    get_trainer,
    init_seed,
)

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, "hyper_tuning/SAS/SAS_BPR_tuning.yaml")]
params_file = os.path.join(current_path, "hyper_tuning/SAS/SAS_hyper_tuning_params_bpr_2.yaml")
output_file = os.path.join(current_path, "hyper_tuning/SAS/SAS_BPR.txt")

def objective_function(config_dict=None, config_file_list=None):

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], False)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model_name = config['model']
    model = get_model(model_name)(config, train_data._dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False)
    test_result = trainer.evaluate(test_data)

    return {
        'model': model_name,
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def tune(algo):
    hp = HyperTuning(
        objective_function,
        algo=algo,
        early_stop=10,
        max_evals=75,
        params_file=params_file,
        fixed_config_file_list=config_file_list,
    )
    hp.run()
    hp.export_result(output_file=output_file)



class TestHyperTuning(unittest.TestCase):
    def test_GRU(self):
        tune(algo="exhaustive")


if __name__ == "__main__":
    unittest.main()