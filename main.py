from settings import params
import torch
import argparse
from env.environment import Environment
from agent.algorithm import make
import train
import sys
sys.path.append(r'./Informer-MVP')
from settings import args as parses
import os


# def parse_args():
#     parser = argparse.ArgumentParser("Experiments for pursuit SUMO environments")
#     parser.add_argument("--exp_name", type=str, default="test", help="addition name of the experiment")  # 实验名
#     parser.add_argument("--port", type=int, default=8813, help="The port of sumo environment")
#     parser.add_argument("--batch_size", type=int, default=32, help="The train batch size")
#     parser.add_argument("--domain_name", type=str, default="3x3Traffic", help="The domain of training")
#     parser.add_argument("--alg_name", type=str, default="PPO", help="The algorithm name of training")
#     parser.add_argument("--reload_exp", type=str, default=None, help="The reload exp name")
#     parser.add_argument("--test", action="store_true", default=False, help="Test")
#     parser.add_argument("--gui", action="store_true", default=False, help="Use gui")
#
#     return parser.parse_args()

# parses = parse_args()

if parses.domain_name == "3x3":
    params["rou_path"] = "./env/3x3/3x3Grid.rou.xml"
    params["cfg_path"] = "./env/3x3/3x3Grid.sumocfg"
    params["net_path"] = "./env/3x3/3x3Grid.net.xml"
elif parses.domain_name == "3x3Traffic":
    params["memory_capacity"] = 20000
    params["rou_path"] = "./env/3x3Traffic/3x3Grid.rou.xml"
    params["cfg_path"] = "./env/3x3Traffic/3x3Grid.sumocfg"
    params["net_path"] = "./env/3x3Traffic/3x3Grid.net.xml"
    params["num_pursuit"] = 4
    params["pursuit_ids"] = ["p0", "p1", "p2", "p3"]
    params["num_evader"] = 2
    params["evader_ids"] = ["e0", "e1"]
    params["code_length"] = 6
    params["num_action"] = 3
    params["history_length"] = 4
    params["local_observation_shape"] = (params["num_pursuit"], 25+8)
    params["global_observation_shape"] = (params["history_length"], params["num_pursuit"], 25+8)
elif parses.domain_name == "4x4Traffic":
    params["memory_capacity"] = 240000
    params["rou_path"] = "./env/4x4Traffic/4x4Grid.rou.xml"
    params["cfg_path"] = "./env/4x4Traffic/4x4Grid.sumocfg"
    params["net_path"] = "./env/4x4Traffic/4x4Grid.net.xml"
    params["num_pursuit"] = 4
    params["pursuit_ids"] = ["p0", "p1", "p2", "p3"]
    params["num_evader"] = 2
    params["evader_ids"] = ["e0", "e1"]
    params["code_length"] = 7
    params["num_action"] = 3
    params["local_observation_shape"] = (params["num_pursuit"] + params["num_evader"], 4*params["code_length"] + 1 + 80)
    params["global_observation_shape"] = (params["num_pursuit"] + params["num_evader"], 4*params["code_length"] + 1 + 80)
elif parses.domain_name == "4x4TrafficSmall":
    params["memory_capacity"] = 100000
    params["rou_path"] = "./env/4x4TrafficSmall/4x4Grid.rou.xml"
    params["cfg_path"] = "./env/4x4TrafficSmall/4x4Grid.sumocfg"
    params["net_path"] = "./env/4x4TrafficSmall/4x4Grid.net.xml"
    params["num_pursuit"] = 4
    params["pursuit_ids"] = ["p0", "p1", "p2", "p3"]
    params["num_evader"] = 2
    params["evader_ids"] = ["e0", "e1"]
    params["code_length"] = 7
    params["num_action"] = 3
    params["local_observation_shape"] = (params["num_pursuit"] + params["num_evader"], 4*params["code_length"] + 1 + 80)
    params["global_observation_shape"] = (params["num_pursuit"] + params["num_evader"], 4*params["code_length"] + 1 + 80)
elif parses.domain_name == "4x4TrafficSmallNoBack":
    params["memory_capacity"] = 100000
    params["rou_path"] = "./env/4x4TrafficSmallNoBack/4x4Grid.rou.xml"
    params["cfg_path"] = "./env/4x4TrafficSmallNoBack/4x4Grid.sumocfg"
    params["net_path"] = "./env/4x4TrafficSmallNoBack/4x4Grid.net.xml"
    params["num_pursuit"] = 4
    params["pursuit_ids"] = ["p0", "p1", "p2", "p3"]
    params["num_evader"] = 2
    params["evader_ids"] = ["e0", "e1"]
    params["code_length"] = 7
    params["num_action"] = 3
    params["local_observation_shape"] = (params["num_pursuit"] + params["num_evader"], 4*params["code_length"] + 1 + 80)
    params["global_observation_shape"] = (params["num_pursuit"] + params["num_evader"], 4*params["code_length"] + 1 + 80)

params["port"] = parses.port
params["gui"] = parses.gui
params["domain_name"] = parses.domain_name
params["algorithm_name"] = parses.alg_name
params["exp_name"] = parses.exp_name
params["batch_size"] = parses.batch_size
params["use_pre"] = parses.use_predict
params["pre_format"] = parses.pre_format
params["assign_method"] = parses.assign_method
params["no_task"] = parses.no_task
params["pre_method"] = parses.pre_method
params["ktr"] = parses.ktr
params["use_ml"] = parses.use_mau_loss
params["use_encoder"] = parses.use_encoder
params["use_adj"] = parses.use_adj
params["directory"] = "output/{}-domain-{}-{}".format(params["domain_name"], params["algorithm_name"], params["exp_name"])

params["env"] = Environment(params)
controller = make(params["algorithm_name"], params)
if params["train_pur"]:
    if params["evader_method"] == "Command" and parses.q_table_path is not None:
        controller_eva = make("Q-Learning", params)
        reload_csv_path = os.path.join("output", parses.q_table_path, "best.csv")
        controller_eva.load_weights_from_history(reload_csv_path)
        print("Load EVA Model success from", reload_csv_path)
        params["Eva_Controller"] = controller_eva

if parses.test:
    assert parses.model_name is not None, "Without the model to load!"
    if params["algorithm_name"] == "Q-Learning":
        reload_model_path = os.path.join("output", parses.model_name, "best.csv")
    else:
        reload_model_path = os.path.join("output", parses.model_name, "best.pth")
    if os.path.exists(reload_model_path):
        params["test_output_csv"] = os.path.join("output", parses.model_name, "best_res.csv")
        controller.load_weights_from_history(reload_model_path)
        print("Load model success form", reload_model_path)
        result = train.test(controller, params, 50)
    else:
        print("Not found model!")
else:
    result = train.run(controller, params)








