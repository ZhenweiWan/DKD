import copy

from dassl.engine import build_trainer
from dassl.utils import setup_logger, set_random_seed, collect_env_info

import torch



def aggregation_model_parameter(clients_model, mode="all"):
    aggregation_parameter = {}
    model_dict = copy.deepcopy(clients_model[0].get_learnable_Parameters())
    for index, (parameter_name) in enumerate(model_dict):

        if mode == "VPT" and "VPT" not in parameter_name:
            continue
        elif mode == "prompt" and "prompt" not in parameter_name:
            continue

        temp_tensor = 0
        for i in range(len(clients_model)):
            try:
                temp_tensor = temp_tensor + clients_model[i].get_learnable_Parameters()[parameter_name]
            except:
                print(i, "miss", parameter_name)
        aggregation_parameter[parameter_name] = temp_tensor / len(clients_model)
    # print(aggregation_parameter.values())
    return aggregation_parameter


def aggregation_model_parameter_weight_avg(clients_model, clients_data_len):
    total_len = sum(clients_data_len)
    aggregation_parameter = {}
    model_dict = copy.deepcopy(clients_model[0].get_learnable_Parameters())
    for index, (parameter_name) in enumerate(model_dict):
        temp_tensor = 0
        for i in range(len(clients_model)):
            temp_tensor = temp_tensor + clients_model[i].get_learnable_Parameters()[parameter_name] * (clients_data_len[i]/total_len)
        aggregation_parameter[parameter_name] = temp_tensor
    return aggregation_parameter

