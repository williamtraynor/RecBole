# -*- coding: utf-8 -*-
# @Time   : 2022/7/15
# @Author : Gaowei Zhang
# @Email  : zgw15630559577@163.com
import os
import unittest
from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, "GRU_BPR_tuning.yaml")]
params_file = os.path.join(current_path, "GRU_hyper_tuning_params.yaml")
output_file = os.path.join(current_path, "GRU_BPR.txt")

def tune(algo):
    hp = HyperTuning(
        objective_function,
        algo=algo,
        early_stop=10,
        max_evals=100,
        params_file=params_file,
        fixed_config_file_list=config_file_list,
    )
    hp.run()
    hp.export_result(output_file=output_file)


class TestHyperTuning(unittest.TestCase):
    def test_GRU(self):
        tune(algo="bayes")


if __name__ == "__main__":
    unittest.main()
